### 답변 토큰에만 손실 계산하기
개념 : SFT의 학습 신호  
SFT는 이미 사전학습한 모델을 입력 + 모범답안 쌍으로 추가 학습하는 것. RAG처럼 정답 문장을 저장소에 넣는게 아니라, 손실을 역전파해서 가중치를 바꿈.  
예시
|학습에 전달하는 것|손실에서 하는 일|
|---|---|
|질문과 규정|답변을 계산하는 문맥으로 읽음|
|모범답안의 이전 토큰|다음 정답 토큰을 예측하는 문맥|
|모범답안의 다음 토큰과 EOS|실제 예측 목표|

손실 계산에서 빼라는 표시 : -100  
- labels에는 ㅜ언래 각 위치의 정답 토큰 ID가 들어감
- 그런데 프롬프트(질문, 규정) 위치에는 "맞혀야 할 정답"이 없음. 그 자리에 뭔가를 넣어야 하니, 진짜 토큰 ID와 절대 겹치지 않는 음수 값을 씀.
- PyTorch의 cross_entropy는 기본적으로 -100인 위치를 손실에서 무시함.(ignore_index=-100)
|표시|의미|대상|
|---|---|---|
|labels=-100|이 위치에서 정답을 맞치는 손실은 계산하지 않음(입력으로는 읽음)|프롬프트, 패딩|
|attention_mask=0|이 위치를 아예 보지 않음|패딩|

collate : 길이가 다른 문항을 한 배치로  
|텐서|실제 토큰 자리|패딩 자리|
|---|---|---|
|input_ids|토큰 ID|pad_token_id|
|attention_mask|1|0(보지 않음)|
|labels|-100 또는 정답 ID|-100(손실 제외)|

shift : 한 칸 밀어 맞추기  
t번째 위치의 출력(logits)은 다음 토큰(t+1)을 예측. 그래서 비교할 때 logits는 마지막을 빼고, labels는 첫 칸을 뺌.  

| 함수 또는 값 | 입력과 결과 | 구분할 점 |
| --- | --- | --- |
| `apply_chat_template` | 역할과 내용 목록을 문자열이나 토큰 ID로 변환 | `tokenize=True`이면 토큰화하고, `return_dict`와 `return_tensors`가 반환 형태를 정함 |
| `add_generation_prompt=True` | assistant 답변이 시작될 위치까지 입력 구성 | 이미 답변을 포함한 학습 대화에서는 이 수업의 설정을 `False`로 둠 |
| `encode_example` | 대화 레코드에서 `input_ids`, `attention_mask`, `labels` 생성 | 프롬프트 접두사가 일치하는지 먼저 확인하고 답변과 EOS까지 사용 |
| `labels=-100` | 해당 위치의 직접 손실 제외 | 프롬프트를 입력 문맥에서 제거하는 값은 아님 |
| `collate`, `pad_sequence` | 여러 사례를 `[배치, 길이]` 텐서로 묶음 | `batch_first=True`, 입력은 pad ID, attention은 0, labels는 −100으로 패딩 |
### 전체 가중치를 업데이트하며 SFT 시작하기
개념 : 전체 SFT의 메모리  
전체 SFT는 모든 파라미터가 학습대상이라서 가중치뿐 아니라 기울기와 AdamW 상태까지 모두 들고 있어야 함.  

학습 전 응답을 먼저 보관  
학습 뒤 같은 입력, 같은 설정으로 비교하려고 dev 12문항의 응답을 학습 전에 모두 저장.  

TrainingArguments : 학습 실행의 설정표  
TrainingArguments는 모델이나 알고리즘이 아니라 Trainer에 주는 설정표. 데이터를 몇 개씩 처리하고, 언제 가중치를 갱신하며, 평가와 저장을 언제 수행할지 전달하는 설정 객체.  

아래 표는 예시  
|파라미터|의미|
|---|---|
|max_steps|optimizer 갱신의 전페 목표(epoch보다 우선)|
|per_device_train_batch_size|한 번에 처리할 문항 수|
|gradient_accumulation_steps|네 문항의 기울기를 모아 한 번 갱신 -> 유효 배치 4|
|learning_rate|학습률 계획의 기준 크기|
|lr_scheduler_type/warmup_steps|두 갱신 동안 올리고 24까지 선형 감소|
|optim/weight_decay|PyTorch AdamW / 감쇠 계수|
|max_grad_norm|갱신 전 기울기 노름 제한|

### 손실 곡선과 학습 설정 읽기
개념 : 곡선은 "내려갔다"까지만 말해 줌  
손실이 내려가는 것은 정답 토큰을 더 잘 맞히게 됐다는 뜻이지, 업무 답이 좋아졌다는 뜻은 아님.  

train_loss : 최근 학습 구간의 평균  
dev loss : 학습에 쓰지 않은 규정 그룹에 대한 손실  

### 학습을 재개하고 추론용 모델을 저장
Trainer 체크포인트에는 가중치 외에 optimizer, scheduler, 난수 상태와 진행 단계가 들어있음.  

체크 포인트 : 특정 시점의 상태를 저장한 것.