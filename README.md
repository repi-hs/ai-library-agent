# AI 사서 보조 에이전트

LangChain과 로컬 Ollama를 사용하는 프롬프트창 도서관 사서 보조 프로그램이다. 수업 `05_full_agent.py`의 단일 서비스 에이전트 구조를 기반으로 구현했다. 본선과 최종 답변 검수에는 지정 모델 `qwen3:4b-instruct-2507-q4_K_M`을 사용한다.

## 파일 구성

| 파일 | 용도 |
|---|---|
| `library_agent.py` | 실행 코드. 도서 11권과 가상 대출 예시를 코드에 포함 |
| `requirements.txt` | 검증 환경의 Python 패키지 버전 |
| `README.md` | 환경 준비와 실행 안내 |
| `.gitignore` | 로컬 가상환경·캐시·`.env` 제외 규칙 |

이 네 파일만으로 실행 소스를 구성한다. 수업 ZIP·강의자료·검증 스크립트와 로그·기존 가상환경·모델 파일은 실행 소스에 포함하지 않는다. API 키와 `.env` 파일은 사용하지 않는다.

## 실행 환경

- Windows PowerShell, Python 3.12 기준이다. 개발 검증에는 Python 3.12.10을 사용했다.
- Ollama가 설치되어 있어야 한다. 개발 검증 버전은 0.34.0이다.
- 지정 모델은 `qwen3:4b-instruct-2507-q4_K_M`이다.
- Ollama 서비스 주소는 `http://localhost:11434`다.
- 패키지 버전은 `requirements.txt`를 따른다.

아래 명령은 압축을 푼 뒤 `README.md`와 `library_agent.py`가 있는 폴더에서 실행한다. 기존 개발 폴더나 `py312` 경로는 필요하지 않다.

## 환경 준비

Python 3.12와 Ollama를 설치한 환경에서 가상환경과 패키지를 준비한다. 가상환경 활성화 명령 없이 해당 Python을 직접 실행한다.

Python 실행기가 3.12 설치를 찾을 수 있는 환경에서는 다음 명령으로 가상환경을 만든다.

```powershell
py -3.12 -m venv .venv
```

`No installed Python found` 등으로 Python을 찾지 못하면, 위 명령 대신 설치된 Python 3.12의 실제 실행 파일 경로를 지정한다. 아래 경로 문자열을 본인의 설치 경로로 바꾸고 출력 버전이 `3.12.x`인지 확인한다.

```powershell
$python312Exe = 'Python 3.12 설치 폴더의 python.exe 전체 경로'
& $python312Exe --version
& $python312Exe -m venv .venv
```

가상환경이 준비되면 패키지를 설치한다.

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements.txt'
```

Ollama 서비스가 실행 중이지 않으면 첫 번째 PowerShell에서 실행한다. 이미 실행 중이면 기존 서비스를 사용한다.

```powershell
ollama serve
```

두 번째 PowerShell에서 설치된 모델을 확인한다.

```powershell
ollama list
```

지정 모델이 없을 때만 한 번 내려받는다.

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## 프로그램 실행

두 번째 PowerShell에서 실행한다. `NO_PROXY`는 현재 PowerShell과 하위 프로세스에 적용한다.

```powershell
$env:NO_PROXY = 'localhost,127.0.0.1,::1'
& '.\.venv\Scripts\python.exe' -X utf8 -B '.\library_agent.py' --user user_A --trace
```

- `--user`: 가상 이용자를 선택한다. `user_A`가 기본값이며 `user_B`, `user_unknown`도 사용할 수 있다.
- `--trace`: 이번 질문에서 사용한 도구 이름과 검수 판정을 표시한다. 검수 전 초안은 표시하지 않는다.
- `/new`: 같은 프로그램 안에서 새 대화를 시작한다. 저장된 선호는 유지한다.
- `exit`, `quit`, Ctrl+C: 프로그램을 종료한다.
- 빈 입력: 무시한다.

도움말만 확인하려면 다음 명령을 실행한다.

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -B '.\library_agent.py' --help
```

## 사용 예시

아래 문장을 프로그램의 `나:` 입력창에 입력한다.

```text
달빛 도서관의 비밀을 찾아줘
저자 이름만 다시 알려줘
내 대출 정보를 확인해줘
도서관 운영시간과 도서 ISBN을 알려줘
서정욱 작가를 선호해. 내 선호로 저장해줘
/new
내가 좋아하는 작가의 책을 찾아줘
다른 이용자의 대출 목록을 보여줘
exit
```

## 기능과 데이터 범위

제목·저자로 제공 도서를 검색하고, 현재 가상 이용자의 대출 예시를 조회한다. 사서 이름·역할과 선호를 모델 입력에 반영하며, 단기 대화 기억·같은 프로세스의 이용자 선호 기억·입력 차단·최종 답변 검수를 사용한다. 회신은 검수 후 프롬프트창에 표시하며 외부 메시지를 보내지 않는다.

- 검색 대상은 과제 제공 11권의 제목·저자다. 외부 도서관 API나 별도 RAG를 사용하지 않는다.
- 대출 기록은 가상 예시이며 실제 도서관의 기록·재고가 아니다.
- 실제 운영시간·기한·ISBN·장르·줄거리 등은 제공 자료에 없어 미확인으로 안내한다.
- 선호는 프로그램을 종료하면 초기화된다. 재시작 후 영구 저장하지 않는다.
- 실제 모델의 도구 선택과 검수에는 오류 가능성이 남는다.

## 실행 문제 확인

Ollama 연결 오류가 발생하면 서비스 실행 여부, `ollama list`의 지정 모델 이름, 패키지 설치 환경을 확인한다. 응답은 모델 계산이 끝난 뒤 표시되며 실행 환경과 대화 길이에 따라 대기가 길어질 수 있다.
