# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation


import logging
import argparse

import numpy as np
import onnx

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="SSD_Mobilenet_v1 fine-tune examples for image classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained ssd_mobilenet_v1 model on onnx file"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="config yaml path"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )

    args = parser.parse_args()

    model = onnx.load(args.model_path)
    if args.benchmark:
        from lpot.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        results = evaluator()
        for mode, result in results.items():
            acc, batch_size, result_list = result
            latency = np.array(result_list).mean() / batch_size

            print('\n{} mode benchmark result:'.format(mode))
            print('Accuracy is {:.3f}'.format(acc))
            print('Batch size = {}'.format(batch_size))
            print('Latency: {:.3f} ms'.format(latency * 1000))
            print('Throughput: {:.3f} images/sec'.format(batch_size * 1./ latency))

    if args.tune:
        from lpot.experimental import Quantization, common

        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        q_model = quantize()
        q_model.save(args.output_model)
        
        if args.benchmark:
            from lpot.experimental import Benchmark
            evaluator = Benchmark(args.config)
            evaluator.model = common.Model(q_model)
            results = evaluator()
            for mode, result in results.items():
                acc, batch_size, result_list = result
                latency = np.array(result_list).mean() / batch_size

                print('\n quantized model {} mode benchmark result:'.format(mode))
                print('Accuracy is {:.3f}'.format(acc))
                print('Batch size = {}'.format(batch_size))
                print('Latency: {:.3f} ms'.format(latency * 1000))
                print('Throughput: {:.3f} images/sec'.format(batch_size * 1./ latency))