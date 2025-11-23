# ecg-reasoning-benchmark
1. GEM, PULSE 둘 다 돌아감 (different conda environment, different imports (./gem and ./pulse))

2. GEM, PULSE 인퍼런스로 현재 나오는 answer도 post-processing 해야되는 상태 (ex. ###Answer: (a) Yes 혹은 그냥 yes로 나옴)

3. checkpoints 디렉토리 만들어서 (https://github.com/YubaoZhao/ECG-Chat) 에서 ECG-CoCa 체크포인트 다운받아서 넣어줘야됨 (ecg-chat 학회 억셉되서 체크포인트 공개함) 

4. TODO: ECG-Chat, process output  