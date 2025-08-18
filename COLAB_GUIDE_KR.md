# Google Colab에서 GitHub 코드 사용 A-Z 가이드

아래 단계에 따라 이 저장소의 슬라이드 요약 도구를 Google Colab에서 실행하고 결과 문서를 Google Drive에 저장할 수 있습니다.

## A. Colab 준비
1. [Google Colab](https://colab.research.google.com/)에 접속하여 새 노트북을 생성합니다.
2. 상단 메뉴에서 `런타임` → `런타임 유형 변경`을 선택하고, 필요하면 GPU/TPU 대신 "표준"을 선택합니다.

## B. Google Drive 마운트
1. 첫 번째 셀에 아래 코드를 실행하여 Drive를 마운트합니다.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. 인증 링크가 표시되면 계정을 선택하고 권한을 허용합니다.

## C. GitHub 저장소 클론
1. 다음 명령을 실행하여 저장소를 클론합니다.
   ```bash
   !git clone https://github.com/your-account/slide_explainer_app_advanced.git
   %cd slide_explainer_app_advanced
   ```
   > 실제 GitHub 경로로 수정하세요.

## D. 필요한 라이브러리 설치
1. 요구 패키지를 설치합니다.
   ```bash
   !pip install -r requirements.txt
   ```
2. `pdf2image` 사용 시 Poppler가 필요하므로 아래 명령도 실행합니다.
   ```bash
   !apt-get install -y poppler-utils
   ```

## E. OpenAI API 키 설정
1. OpenAI에서 발급받은 API 키를 환경 변수로 설정합니다.
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
   ```

## F. PDF 파일 준비
1. 요약할 PDF를 Google Drive에 업로드합니다. 예시 경로: `/content/drive/MyDrive/lecture.pdf`.

## G. 요약 실행
1. 아래 명령으로 슬라이드 요약을 실행합니다.
   ```bash
   !python -m slide_explainer_app_advanced.main \
     --pdf /content/drive/MyDrive/lecture.pdf \
     --title "Lecture Summary" \
     --drive-dir "/content/drive/MyDrive" \
     --groups 2
   ```
   - `--drive-dir`는 결과 문서를 저장할 Drive 경로입니다.
   - 기타 옵션은 `--help`로 확인할 수 있습니다.

## H. 결과 확인
1. 실행이 끝나면 `drive_dir`에 지정한 위치에 `<title>.txt` 파일이 생성됩니다.
2. Google Drive에서 해당 파일을 열어 요약 내용을 확인합니다.

## Z. 마무리
- 작업이 끝나면 노트북을 종료하거나 Drive 마운트를 해제합니다.

이 가이드는 Colab 환경에서 GitHub 코드를 사용하여 슬라이드 요약을 생성하고 Google Drive에 결과를 저장하는 전체 과정을 설명합니다.
