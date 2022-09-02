import tensorflow as tf
import os
import time
import argparse
import datetime

from model import *
from data_loader import *
from config import *


# for training
def train_model(model, batch_gen, num_train_steps, is_save=0, graph_dir_name='default'):
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    summary = None
    val_summary = None
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        early_stop_count = MAX_EARLY_STOP_COUNT

        writer = tf.summary.FileWriter('./graph/' + graph_dir_name, sess.graph)

        initial_time = time.time()
        min_ce = 1000000
        best_dev_accr = 0
        test_accr_at_best_dev = 0
        for index in range(num_train_steps):
            try:
                # run train
                raw_encoder_inputs, raw_encoder_seq, raw_encoder_prosody, raw_label = batch_gen.get_batch(
                    data=batch_gen.train_set,
                    batch_size=model.batch_size,
                    encoder_size=model.encoder_size,
                    is_test=False
                )

                # prepare data which will be push from pc to placeholder
                input_feed = {}
                input_feed[model.encoder_inputs] = raw_encoder_inputs
                input_feed[model.encoder_seq] = raw_encoder_seq
                input_feed[model.encoder_prosody] = raw_encoder_prosody
                input_feed[model.y_labels] = raw_label
                input_feed[model.dr_prob] = model.dr
                _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
                writer.add_summary(summary, global_step=model.global_step.eval())
            except:
                print("excepetion occurs in train step")
                pass
            # run validation
            dev_ce, dev_accr, dev_summary, _ = run_test(sess=sess,model=model,batch_gen=batch_gen,data=batch_gen.dev_set)
            writer.add_summary(dev_summary, global_step=model.global_step.eval())
            end_time = time.time()
            if index > CAL_ACCURACY_FROM:
                    if (dev_ce < min_ce):
                        min_ce = dev_ce
                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval())
                        early_stop_count = MAX_EARLY_STOP_COUNT
                        test_ce, test_accr, _, _ = run_test(sess=sess,
                                                            model=model,
                                                            batch_gen=batch_gen,
                                                            data=batch_gen.test_set)
                        best_dev_accr = dev_accr
                        test_accr_at_best_dev = test_accr
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print( "early stopped")
                            break
                        test_accr = 0
                        early_stop_count = early_stop_count - 1
                    print(str(int(end_time - initial_time) / 60) + " mins" + \
                    " step/seen/itr: " + str(model.global_step.eval()) + "/ " + \
                    str(model.global_step.eval() * model.batch_size) + "/" + \
                    str(round(model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)) + \
                    "\tdev_accr: " + '{:.3f}'.format(dev_accr) + "  test_accr: " + '{:.3f}'.format(
                        test_accr) + "  loss: " + '{:.2f}'.format(dev_ce))

        writer.close()
        print('Total steps : {}'.format(model.global_step.eval()))
        with open('./Result_audio.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + str(best_dev_accr) + '\t' + str(test_accr_at_best_dev) + '\n')


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def run_test(sess, model, batch_gen, data):
    list_batch_ce = []
    list_batch_correct = []

    list_pred = []
    list_label = []
    #print("len(data)",len(data))
    max_loop = len(data) // model.batch_size
    remaining = len(data) % model.batch_size

    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    #print("max_loop",max_loop)
    for test_itr in range(max_loop + 1):
        raw_encoder_inputs, raw_encoder_seq, raw_encoder_prosody, raw_label = batch_gen.get_batch(
            data=data,
            batch_size=model.batch_size,
            encoder_size=model.encoder_size,
            is_test=True,
            start_index=(test_itr * model.batch_size)
        )

        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.encoder_inputs] = raw_encoder_inputs
        input_feed[model.encoder_seq] = raw_encoder_seq
        input_feed[model.encoder_prosody] = raw_encoder_prosody
        input_feed[model.y_labels] = raw_label
        input_feed[model.dr_prob] = 1.0  # no drop out while evaluating

        try:
            bpred, bloss = sess.run([model.batch_pred, model.batch_loss], input_feed)
        except:
            print("excepetion occurs in valid step : " + str(test_itr))
            pass

        # remaining data case (last iteration)
        if test_itr == (max_loop):
            bpred = bpred[:remaining]
            bloss = bloss[:remaining]
            raw_label = raw_label[:remaining]

        # batch loss
        list_batch_ce.extend(bloss)

        # batch accuracy
        list_pred.extend(np.argmax(bpred, axis=1))
        list_label.extend(np.argmax(raw_label, axis=1))

    list_batch_correct = [1 for x, y in zip(list_pred, list_label) if x == y]

    sum_batch_ce = np.sum(list_batch_ce)
    accr = np.sum(list_batch_correct) / float(len(data))

    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr)
    summary = summary_pb2.Summary(value=[value1, value2])

    return sum_batch_ce, accr, summary, list_pred
def main(data_path, batch_size, encoder_size, num_layer, hidden_dim,
         num_train_steps, lr, is_save, graph_dir_name,
         dr):
    if is_save is 1:
        create_dir('save/')
        create_dir('save/' + graph_dir_name)

    create_dir('graph/')
    create_dir('graph/' + graph_dir_name)

    batch_gen = ProcessDataAudio(data_path)

    model = Encoder_Audio(
        batch_size=batch_size,
        encoder_size=encoder_size,
        num_layer=num_layer,
        lr=lr,
        hidden_dim=hidden_dim,
        dr=dr
    )

    model.build_graph()
    train_model(model, batch_gen, num_train_steps, is_save, graph_dir_name)


if __name__ == '__main__':
    graph_name = "graphe2"
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    main(
        data_path='E:/data/SER/IEMOCAP/processed/four_category/audio_woZ_set01/',
        batch_size=64,
        encoder_size=750,
        num_layer=1,
        hidden_dim=200,
        num_train_steps=10000,
        lr=0.001,
        is_save=0,
        graph_dir_name=graph_name,
        dr=0.7
    )
    '''
        main(
            data_path=args.data_path,
            batch_size=args.batch_size,
            encoder_size=args.encoder_size,
            num_layer=args.num_layer,
            hidden_dim=args.hidden_dim,
            num_train_steps=args.num_train_steps,
            lr=args.lr,
            is_save=args.is_save,
            graph_dir_name=graph_name,
            dr=args.dr
        )
        '''