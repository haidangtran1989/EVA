import tensorflow as tf
from transformers import TFAutoModel
from official.nlp import optimization
from tensorflow.keras.layers import Dot, Dense, Activation
from tensorflow.keras import mixed_precision
from utils.config import *

# Use mixed precision to speed up
mixed_precision.set_global_policy('mixed_float16')

# Create a MirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = LOCAL_BATCH_SIZE * strategy.num_replicas_in_sync

with strategy.scope():
    # Topic aware sampling BERT model
    encoder = TFAutoModel.from_pretrained(BASE_MODEL_LINK, from_pt=True)
    entity_linear_dense = Dense(ENTITY_DIMENSION, activation=None, name='entity_linear_ffn')
    knrm_linear_dense = Dense(KERNEL_DIMENSION_OUTPUT, activation=None, name='knrm_linear_ffn')
    linear_activation = Activation('linear', dtype=tf.float32)
    tanh_activation = Activation('tanh', dtype=tf.float32)
    similarity = Dot(axes=1, normalize=False, dtype=tf.float32)


    def build_input_layer(max_len, part):
        input_ids_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name=(part + "_ids"))
        input_mask_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name=(part + "_mask"))
        entity_input_layer = tf.keras.layers.Input(shape=(ENTITY_DIMENSION,), dtype=tf.float16,
                                                   name=(part + "_entity_input"))
        if USE_KNRM and part != "query":
            knrm_layer = tf.keras.layers.Input(shape=(KERNEL_DIMENSION_INPUT,), dtype=tf.float16, name=(part + "_knrm"))
            return input_ids_layer, input_mask_layer, entity_input_layer, knrm_layer
        else:
            return input_ids_layer, input_mask_layer, entity_input_layer


    def get_text_embedding(text_ids, text_attention_mask):
        sequence_output = encoder(text_ids, attention_mask=text_attention_mask)["last_hidden_state"]
        return sequence_output[:, 0, :]


    def get_vector_representation(text_ids, text_attention_mask, entity_input):
        bert_textual_output = get_text_embedding(text_ids, text_attention_mask)
        pooled_output = tf.concat([linear_activation(bert_textual_output),
                                   linear_activation(entity_linear_dense(entity_input))], 1)
        return pooled_output


    def pairwise_hinge_loss(actual, prediction):
        positive_passage_score = tf.gather(prediction, [0], axis=1)
        negative_passage_score = tf.gather(prediction, [1], axis=1)
        return tf.reduce_sum(tf.maximum(0.0, 1 - positive_passage_score + negative_passage_score)) / global_batch_size


    def build_optimizer(training_size):
        steps_per_epoch = training_size / global_batch_size
        num_train_steps = steps_per_epoch * TRAINING_EPOCHS
        num_warmup_steps = int(0.03 * num_train_steps)
        return optimization.create_optimizer(init_lr=INITIAL_LEARNING_RATE,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             optimizer_type="adamw")


    def build_ranking_model(training_size):
        query_ids, query_mask, query_entity = build_input_layer(MAX_QUERY_LEN, "query")
        if USE_KNRM:
            pos_passage_ids, pos_passage_mask, pos_passage_entity, pos_knrm = build_input_layer(MAX_PASSAGE_LEN, "positive_passage")
            neg_passage_ids, neg_passage_mask, neg_passage_entity, neg_knrm = build_input_layer(MAX_PASSAGE_LEN, "negative_passage")
        else:
            pos_passage_ids, pos_passage_mask, pos_passage_entity = build_input_layer(MAX_PASSAGE_LEN, "positive_passage")
            neg_passage_ids, neg_passage_mask, neg_passage_entity = build_input_layer(MAX_PASSAGE_LEN, "negative_passage")
        query_representation = get_vector_representation(query_ids, query_mask, query_entity)
        positive_passage_representation = get_vector_representation(pos_passage_ids, pos_passage_mask,
                                                                    pos_passage_entity)
        negative_passage_representation = get_vector_representation(neg_passage_ids, neg_passage_mask,
                                                                    neg_passage_entity)
        if USE_KNRM:
            pos_knrm_score = tanh_activation(knrm_linear_dense(pos_knrm))
            neg_knrm_score = tanh_activation(knrm_linear_dense(neg_knrm))
        positive_relevance = similarity([query_representation, positive_passage_representation])
        negative_relevance = similarity([query_representation, negative_passage_representation])
        relevance = tf.concat([positive_relevance, negative_relevance], 1)
        if USE_KNRM:
            knrm_relevance = tf.concat([pos_knrm_score, neg_knrm_score], 1)
            relevance = relevance + knrm_relevance
            model_to_rank = tf.keras.Model(inputs=[query_ids, query_mask, query_entity,
                                                   pos_passage_ids, pos_passage_mask, pos_passage_entity, pos_knrm,
                                                   neg_passage_ids, neg_passage_mask, neg_passage_entity, neg_knrm],
                                           outputs=relevance)
        else:
            model_to_rank = tf.keras.Model(inputs=[query_ids, query_mask, query_entity,
                                                   pos_passage_ids, pos_passage_mask, pos_passage_entity,
                                                   neg_passage_ids, neg_passage_mask, neg_passage_entity],
                                           outputs=relevance)
        model_to_rank.compile(loss=pairwise_hinge_loss, optimizer=build_optimizer(training_size))
        return model_to_rank
