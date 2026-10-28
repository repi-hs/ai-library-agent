# AI 사서 보조 에이전트

수업 `05_full_agent.py`를 기반으로 LangChain과 로컬 Qwen3를 활용한 도서관 사서 보조 에이전트다.

## 주요 기능

- 제공된 11권의 도서를 제목·저자로 검색한다.
- 현재 이용자의 가상 대출 정보를 조회한다.
- Runtime Context와 미들웨어로 사서 정보와 이용자 선호를 모델 입력에 반영한다.
- 단기 기억으로 같은 대화의 검색 결과와 후속 질문을 연결한다.
- 이용자 선호를 Store에 저장하고, 프로그램 실행 중 새 대화(`/new`)에서도 검색에 활용한다.
- 비업무·타인 비공개 정보 요청을 차단하고, 최종 답변을 도구 결과와 대조해 검수한다.
- 과제 자료에 없는 운영시간·대출 기한·ISBN 등은 미확인으로 안내한다.

## 실행 환경

- Windows PowerShell, Python 3.12, Ollama
- 모델: `qwen3:4b-instruct-2507-q4_K_M`
- Python 패키지: `requirements.txt`

## 설치

저장소를 내려받은 폴더에서 실행한다.

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements.txt'
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

`py -3.12`가 Python을 찾지 못하면 설치된 Python 3.12의 `python.exe` 경로를 지정해 가상환경을 생성한다.

## 실행

Ollama가 실행 중이지 않으면 별도 PowerShell에서 시작한다.

```powershell
ollama serve
```

프로젝트 폴더의 PowerShell에서 에이전트를 실행한다.

```powershell
$env:NO_PROXY = 'localhost,127.0.0.1,::1'
& '.\.venv\Scripts\python.exe' -X utf8 -B '.\library_agent.py' --user user_A --trace
```

- `--user`: 가상 이용자 선택 (`user_A`, `user_B`, `user_unknown`)
- `--trace`: 사용한 도구와 검수 판정 표시
- `/new`: 새 대화 시작
- `exit`, `quit`, Ctrl+C: 종료

## 사용 예시

```text
달빛 도서관의 비밀을 찾아줘
저자 이름만 다시 알려줘
내 대출 정보를 확인해줘
서정욱 작가를 선호해. 내 선호로 저장해줘
/new
내가 좋아하는 작가의 책을 찾아줘
다른 이용자의 대출 목록을 보여줘
exit
```

## 파일 구성

- `library_agent.py`: 에이전트 실행 코드
- `requirements.txt`: 패키지 버전
- `README.md`: 설치·실행 안내
- `.gitignore`: 가상환경·캐시 제외 규칙
