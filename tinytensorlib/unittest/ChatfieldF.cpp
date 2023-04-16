/* Copyright EPFL 2023
 * Joshua Klein
 *
 * Implementation of Chatfield-F variant CNN based on VGG8 [1].
 *
 * [1] Chatfield, Ken, et al. "Return of the devil in the details: Delving deep
 * into convolutional nets." arXiv preprint arXiv:1405.3531 (2014).
 *
 */

#include <iostream>
#include "../../taskflow/taskflow/algorithm/pipeline.hpp"
#include "../../taskflow/taskflow/taskflow.hpp"
#include "../tinytensorlib.hh"
#include "ChatfieldF.hh"
#include "Chatfield_layers.hh"

using namespace std;


int
main(int argc, char * argv[])
{
    // Initialize buffers and other vars.
    int sys_info = 0;

    generatePingPongOutputDataStructure(conv1);
    printLayerInfo(&conv1);
    
    connectLayers(pool1, conv1);
    connectLayers(conv2, pool1);
    connectLayers(pool2, conv2);
    connectLayers(conv3, pool2);
    connectLayers(conv4, conv3);
    connectLayers(conv5, conv4);
    connectLayers(pool3, conv5);
    connectLayers(flatten1, pool3);
    connectLayers(dense1, flatten1);
    connectLayers(dense2, dense1);

    dense3.input = dense2.output;
    printLayerInfo(&dense3);

    // Do inference.
    cout << "Starting inference...\n";
    unsigned int idx_arr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int inf_in = 0;
    unsigned int inf_out = 0;

    tf::Executor executor(8);
    tf::Taskflow taskflow;

    tf::Pipeline pl(8, // Maximum level of parallelism, a.k.a., number of threads.
                    // Thread 0 pipe.
                    tf::Pipe{
        tf::PipeType::SERIAL,
        [&idx_arr, &inf_in] (tf::Pipeflow & pf) {
            if (inf_in == warmup_infs) {
                system("m5 resetstats");
            }

            // Stop pipeline if all inferences performed.
            if (pf.token() == T_x) {
                pf.stop();
            } else {
                thread0Work(idx_arr[0], inf_in);
                idx_arr[0] = (idx_arr[0] + 1) % 2;
                inf_in++;
            }
        }
        },
        // Thread 1 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread1Work(idx_arr[1]);
            idx_arr[1] = (idx_arr[1] + 1) % 2;
        }
        },
        // Thread 2 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread2Work(idx_arr[2]);
            idx_arr[2] = (idx_arr[2] + 1) % 2;
        }
        },
        // Thread 3 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread3Work(idx_arr[3]);
            idx_arr[3] = (idx_arr[3] + 1) % 2;
        }
        },
        // Thread 4 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread4Work(idx_arr[4]);
            idx_arr[4] = (idx_arr[4] + 1) % 2;
        }
        },
        // Thread 5 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread5Work(idx_arr[5]);
            idx_arr[5] = (idx_arr[5] + 1) % 2;
        }
        },
        // Thread 6 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr] (tf::Pipeflow & pf) {
            thread6Work(idx_arr[6]);
            idx_arr[6] = (idx_arr[6] + 1) % 2;
        }
        },
        // Thread 7 pipe.
        tf::Pipe{
        tf::PipeType::PARALLEL,
        [&idx_arr, &inf_out] (tf::Pipeflow & pf) {
            thread7Work(idx_arr[7], inf_out);
            idx_arr[7] = (idx_arr[7] + 1) % 2;
            inf_out++;
            cout << "Finished Inference " << inf_out << "!\n";

            if (inf_out == warmup_infs + roi_infs) {
                system("m5 exit");
            }
        }
    }
    );

    tf::Task pipeline = taskflow.composed_of(pl);
    executor.run_n(taskflow, T_x);
    executor.wait_for_all();

    printVector(dense3.output[T_x-1], 5);

    delete[] conv1.input;
    delete[] pool1.input;
    delete[] conv2.input;
    delete[] pool2.input;
    delete[] conv3.input;
    delete[] conv4.input;
    delete[] conv5.input;
    delete[] pool3.input;
    delete[] flatten1.input;
    delete[] dense1.input;
    delete[] dense2.input;
    delete[] dense3.input;
    delete[] dense3.output;

    return 0;
}
