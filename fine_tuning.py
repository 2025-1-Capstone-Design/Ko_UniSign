"""
파일: fine_tuning.py
설명: Uni-Sign 학습시 Wandb 로깅 추가

작성자: 김도완 <dowan.test@gamail.com>
생성일: 2025-04-15
최종 수정일: 2025-04-15
버전: 1.0.0

변경 내역:
- 2025-04-15: Wandb 로깅 추가 (김도완)
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets_ours import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

import wandb

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    # --- 2. Wandb Initialization (Main Process Only) ---
    if utils.is_main_process():
        # 프로젝트 이름, 실행 이름 등을 설정할 수 있습니다.
        run_name = f"{args.dataset}_{args.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="Uni-Sign-Finetuning", # <-- 원하는 프로젝트 이름으로 변경
            name=run_name,
            config=vars(args), # 하이퍼파라미터를 config에 저장
            mode="online" if args.wandb_online else "disabled" # wandb 활성화/비활성화 제어
        )
    # ----------------------------------------------------

    print(f"Creating dataset:")
        
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                             args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
    
    dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], 
                           args=args, phase='dev')
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler, 
                                pin_memory=args.pin_mem)
        
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], 
                            args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = Uni_Sign(
                args=args
                )
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
    
        # --- 수정된 부분 ---
        # 현재 모델의 state_dict 가져오기 (키 존재 및 크기 비교용)
        current_model_dict = model.state_dict()
        
        # 제외할 키 목록 (크기 불일치 발생 키)
        keys_to_exclude = []
        # 로드할 state_dict를 순회하며 현재 모델과 크기가 다른 키 찾기
        for k, v in state_dict.items():
            if k in current_model_dict:
                if current_model_dict[k].shape != v.shape:
                    keys_to_exclude.append(k)
                    print(f"Excluding key due to size mismatch: {k}. Checkpoint shape: {v.shape}, Model shape: {current_model_dict[k].shape}")
            # else: # 체크포인트에는 있지만 모델에는 없는 키 (unexpected key) - strict=False가 처리해줌
    
        # 원본 state_dict를 수정하지 않고, 제외할 키를 뺀 새로운 dict 생성
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_exclude}
        
        if keys_to_exclude:
            print(f"Total {len(keys_to_exclude)} keys excluded due to size mismatch.")
    
        # 필터링된 state_dict를 사용하여 로드 (strict=False는 여전히 유효)
        ret = model.load_state_dict(filtered_state_dict, strict=False)
        # ---------------

        # print('Missing keys: \n', '\n'.join(ret.missing_keys))
        # print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    
    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR":
                print("📄 dev result")
                evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
                # --- Log evaluation metrics to wandb ---
                if wandb.run is not None:
                    log_dict_dev = {f'eval_dev/{k}': v for k, v in dev_stats.items()}
                    wandb.log(log_dict_dev, step=0) # step 0 또는 적절한 값
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
            # --- Log evaluation metrics to wandb ---
            if wandb.run is not None:
                log_dict_test = {f'eval_test/{k}': v for k, v in test_stats.items()}
                wandb.log(log_dict_test, step=0)
        # --- 4. Finish Wandb Run ---
        if utils.is_main_process() and wandb.run is not None:
            wandb.finish()
        return 
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        print(f"epoch {epoch}")
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        # --- 3. Log training metrics to wandb (Main Process Only) ---
        if utils.is_main_process() and wandb.run is not None:
            log_dict_train = {f'train/{k}': v for k, v in train_stats.items()}
            log_dict_train['epoch'] = epoch
            # 현재 learning rate 로깅 (DeepSpeed 사용 시 optimizer 구조 확인 필요)
            current_lr = optimizer.param_groups[0]['lr']
            log_dict_train['train/learning_rate'] = current_lr
            wandb.log(log_dict_train) # 기본적으로 global step 사용, epoch 기준으로 하려면 commit=False 후 별도 로그
        # -------------------------------------------------------------
        
        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)
            
            # adapter_save_dir = f"{output_dir}/'checkpoint_apapter_{epoch}'" # 어댑터 저장 디렉토리
            # uni_sign_weights_save_path = f"{output_dir}/'checkpoint_uni_sign_{epoch}.pth'" # Uni-Sign 자체 가중치 파일
        
            # # 1. LoRA 어댑터 저장
            # model_without_ddp.lora_model.save_pretrained(adapter_save_dir)
            # model_without_ddp.gemma_tokenizer.save_pretrained(adapter_save_dir)
            # print(f"LoRA adapter saved to {adapter_save_dir}")
        
            # # 2. Uni-Sign 자체 가중치 저장 (Gemma 및 LoRA 제외)
            # uni_sign_state_dict = {}
            # # lora_model 내부 파라미터 이름 접두사 확인 (예: 'lora_model.')
            # lora_prefix = "lora_model."
            # # 또는 Gemma 모델 파라미터 이름 접두사 확인 (예: 'gemma_model.') - Uni_Sign 구조에 따라 다름
            # # gemma_prefix = "gemma_model." # 혹은 lora_model 내부의 base_model 접근 경로
        
            # for name, param in model_without_ddp.named_parameters():
            #     # LoRA 모델(PeftModel) 또는 그 내부의 베이스 모델 파라미터가 아니면 저장
            #     if not name.startswith(lora_prefix): # Uni_Sign 내 lora_model 객체 이름 기준
            #        # 만약 gemma_model도 별도 속성으로 있다면 그것도 제외
            #        # if not name.startswith(gemma_prefix): # Uni_Sign 내 gemma_model 객체 이름 기준
            #        uni_sign_state_dict[name] = param.cpu().clone() # CPU로 복사하여 저장
        
            # if uni_sign_state_dict: # 저장할 가중치가 있다면
            #     utils.save_on_master({'uni_sign_weights': uni_sign_state_dict}, uni_sign_weights_save_path)
            #     print(f"Uni-Sign specific weights saved to {uni_sign_weights_save_path}")
            # else:
            #     print("No Uni-Sign specific weights found to save (excluding LoRA/Gemma).")

        # single gpu inference
        if utils.is_main_process():
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            # evaluate(args, test_dataloader, model, model_without_ddp, phase='test')    # 테스트셋 검증 제거

            # --- 3. Log evaluation metrics to wandb ---
            if wandb.run is not None:
                log_dict_eval = {}
                for k, v in test_stats.items():
                    log_dict_eval[f'dev/{k}'] = v
                # for k, v in test_stats.items():
                #     log_dict_eval[f'test/{k}'] = v
                # epoch 기준으로 로깅 (wandb step과 별개로 관리 가능)
                log_dict_eval['epoch'] = epoch
                wandb.log(log_dict_eval)
            # ------------------------------------------
            
            if args.task == "SLT":
                if max_accuracy < test_stats["bleu4"]:
                    max_accuracy = test_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > test_stats["wer"]:
                    max_accuracy = test_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)
                            
                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {test_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # --- 4. Final Evaluation on Test Set (after loop) ---
    if utils.is_main_process():
        print("\n--- Training Finished ---")
        print("Loading best model checkpoint for final test evaluation...")
        best_checkpoint_path = output_dir / 'best_checkpoint.pth'

        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')

            # state_dict 추출 (체크포인트 구조에 따라 키 확인)
            if 'model' in checkpoint:
                best_state_dict = checkpoint['model']
            else:
                best_state_dict = checkpoint # 파일 자체가 state_dict 일 경우

            # Load best weights into the model
            # get_requires_grad_dict 로 저장했으므로 strict=False 사용 권장
            load_result = model_without_ddp.load_state_dict(best_state_dict, strict=False)
            print(f"Loaded best checkpoint from {best_checkpoint_path}")
            # 로드 결과 상세 출력 (선택 사항)
            # print(f"  Missing keys count: {len(load_result.missing_keys)}")
            # print(f"  Unexpected keys count: {len(load_result.unexpected_keys)}")

            print("\n--- Evaluating on Test Set with Best Model ---")
            # evaluate 함수 호출
            final_test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

            print("\n--- Final Test Set Performance (Best Model based on Dev Set) ---")
            # 결과 출력
            for key, value in final_test_stats.items():
                 # 소수점 아래 4자리까지 포맷팅하여 출력
                 print(f"  {key}: {value:.4f}")

            # Log final test stats to wandb
            if wandb.run is not None:
                final_log_dict = {f'final_test/{k}': v for k, v in final_test_stats.items()}
                wandb.log(final_log_dict)
                # wandb 요약(Summary) 업데이트 -> 대시보드에서 최종 값 보기 편함
                wandb.summary.update(final_log_dict)
                wandb.summary.update({'best_dev_metric': max_accuracy, 'best_epoch': epoch}) # 어떤 기준으로 best 모델이 나왔는지 기록

            # Log final test stats to a file
            if args.output_dir:
                with (output_dir / "final_results.txt").open("w") as f:
                    f.write("Final Test Set Performance (Best Model based on Dev Set):\n")
                    f.write(json.dumps(final_test_stats, indent=4) + "\n") # 보기 좋게 indent 추가
                    f.write(f"\nBest model loaded from: {best_checkpoint_path}\n")
                    # 어떤 dev metric 기준으로 best 인지 명시
                    metric_name = "metric"
                    if args.task == "SLT": metric_name = "BLEU4"
                    elif args.task == "ISLR": metric_name = "PI Acc"
                    elif args.task == "CSLR": metric_name = "WER"
                    f.write(f"Achieved best Dev {metric_name}: {max_accuracy:.4f}\n")

        else:
            print(f"Warning: Best checkpoint not found at {best_checkpoint_path}. Skipping final test evaluation.")
    # ----------------------------------------------------

    # --- Finish Wandb Run ---
    # 최종 평가 결과까지 로깅 후 종료
    if utils.is_main_process() and wandb.run is not None:
        wandb.finish()
    # ---------------------------

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if step==30:
        #     break
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(src_input, 
                                                max_new_tokens=100, 
                                                num_beams = 4,
                        )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])
                ## ----------Debug용 출력값 확인
                # tgt_pre = model_without_ddp.gemma_tokenizer.decode(output[i], skip_special_tokens=True)
                # tgt_ref = tgt_input['gt_sentence'][i]
                # print(f"Pred: '{tgt_pre}', Ref: '{tgt_ref}', Match: {tgt_pre == tgt_ref}")

    tokenizer = model_without_ddp.gemma_tokenizer
    # 값이 eos_token_id로 설정 되어있음
    padding_value = tokenizer.eos_token_id
    
    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    
    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)