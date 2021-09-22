## original check:
batch_size = 8
eval_accuracy = 0.8627
throughput = 37.713

You need to train the pretrained-model before evaluation,
```
cd neural-compressor/examples/pytorch/eager/huggingface_models

export TASK_NAME=MRPC
python examples/text-classification/run_glue_tune.py  --model_name_or_path roberta-base   --task_name $TASK_NAME   --do_train  --do_eval  --max_seq_length 128  --per_device_train_batch_size 32   --learning_rate 2e-5   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir

```
Alternately, you could also specify --model_name_or_path to the directory of local .bin model to skip training.
```
python examples/text-classification/run_glue_tune.py  --model_name_or_path /tmp/MRPC   --task_name $TASK_NAME   --do_eval  --max_seq_length 128  --per_device_train_batch_size 32   --learning_rate 2e-5   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir
```



## Lpot fine-tune:
batch_size = 8
Accuracy = 0.89076
throughput = 40.831

https://github.com/leonardozcm/neural-compressor/tree/master/examples/pytorch/eager/huggingface_models

```
bash run_tuning.sh --topology=xlm-roberta-base_MRPC --dataset_location=/root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad --input_model=/tmp/$TASK_NAME/


python examples/text-classification/run_glue_tune.py --tuned_checkpoint saved_results --task_name MRPC --max_seq_length 128 --benchmark --int8 --output_dir /tmp/$TASK_NAME/ --model_name_or_path roberta-base
```


## Lpot fine-tune + prune:

pruning takes time > 15h

```
python examples/text-classification/run_glue_no_trainer_prune.py --task_name mnli --max_length 128 \
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured \
       --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3 --output_dir /tmp/$TASK_NAME/ \
       --prune --config prune.yaml --output_model prune_model/model.pt --seed 5143
```

## ONNX:
batch_size = 8
accuracy = 0.8627
throughput = 64.303

You need to [prepare dataset in advance](https://github.com/intel/neural-compressor/tree/master/examples/onnxrt/language_translation/roberta#prepare-dataset), place it under ./examples/text-classification/MRPC like described in  ./examples/text-classification/bert.yaml

refer to https://github.com/intel/neural-compressor/tree/master/examples/onnxrt/language_translation/roberta
```
python examples/text-classification/run_glue_tune.py --model_name_or_path /tmp/MRPC  --task_name MRPC --max_seq_length 128  --output_dir /tmp/$TASK_NAME/ --eval_onnx
```

+ quantize and optimize
```
python examples/text-classification/run_glue_tune.py --model_name_or_path /tmp/MRPC  --task_name MRPC --max_seq_length 128  --output_dir /tmp/$TASK_NAME/ --eval_onnx --onnx_quantize
```
batch_size = 8
accuracy =0.8725
throughput = 80.140

+ 4 instance and 12 core per instance(need to config in bert.yaml)
```
    configs:
      cores_per_instance: 12
      num_of_instance: 4
```
batch_size = 8
accuracy = 0.8725
throughput = 472.605 (214.663 without quanization and optimization)


## bigdl-nano (jemalloc + omp):
batch_size = 8
accuracy = 0.8627
throughput = 55.658
```
bigdl-nano-init python examples/text-classification/run_glue_tune.py  --model_name_or_path /tmp/MRPC   --task_name $TASK_NAME   --do_eval  --max_seq_length 128  --per_device_train_batch_size 32   --learning_rate 2e-5   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir
```

4 pinned Multiprocessing:
eval_accuracy = 0.8627
throughput = 106.188
```
num_multiprocessing=4 python examples/text-classification/run_glue_tune.py  --model_name_or_path /tmp/MRPC   --task_name $TASK_NAME   --do_eval  --max_seq_length 128  --per_device_train_batch_size 32   --learning_rate 2e-5   --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir
```
