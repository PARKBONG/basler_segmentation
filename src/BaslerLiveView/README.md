# BaslerLiveView

Basler pylon 카메라를 연결해 실시간 영상을 WPF 창에 띄우는 최소 예제.

## 구성

| 파일 | 역할 |
|------|------|
| `CameraService.cs` | pylon `Camera` 래퍼 — 연결/연속 그랩/`PixelDataConverter`(→BGRA) |
| `FrameCropper.cs` | 크롭 창 계산/실행 (위치는 축별 0~100%) — 세그멘테이션 입력 + 화면 오버레이 좌표 |
| `FrameRecorder.cs` | Record 버튼 저장 — 저장 fps 샘플링 + 큐 기반 백그라운드 PNG 인코딩/기록 |
| `MainWindow.xaml(.cs)` | 카메라 선택·연결·시작/정지 UI, `WriteableBitmap`으로 프레임 표시 |
| `App.xaml.cs` | `--emulate` 인자로 소프트웨어 카메라 에뮬레이터 활성화 |
| `BaslerLiveView.csproj` | `Basler.Pylon.dll` 참조 + `pylon\Runtime\x64` 네이티브 DLL을 출력 폴더로 복사 |

## 동작 흐름

```
CameraFinder.Enumerate()                    // 카메라 열거
  → new Camera(info) + Configuration.AcquireContinuous
  → Open()
  → StreamGrabber.Start(LatestImages, ProvidedByStreamGrabber)
  → ImageGrabbed 이벤트 (grab 스레드)
      → PixelDataConverter.Convert → BGRA byte[]
      → FrameRecorder.Submit         // Record 중이고 저장 주기가 됐으면 '풀사이즈 원본'을 큐에 복사
      → (Segmentation 켠 경우)
          FrameCropper.Crop          // 크롭본 = 모델 입력
          → SegmentationService.Submit
          → Compose()                // 풀 프레임 위 크롭 위치에 최신 추론 결과를 다시 얹음
      → Dispatcher.Invoke → WriteableBitmap.WritePixels  // 항상 '풀 프레임'을 표시
```

화면은 언제나 풀 프레임이고, 크롭 영역은 그 위에 빨간 네모(`OverlayCanvas`)로 표시만 됩니다.
바깥이 보여야 프레임을 어디로 옮길지 판단할 수 있기 때문입니다. 모델은 학습 때와 같은 화각을
보도록 크롭본만 받고, 그 결과 오버레이는 네모 안쪽에 되돌려 그립니다.

- 그랩은 pylon이 제공하는 별도 스레드에서 돌고, 프레임마다 `ImageGrabbed`가 발생합니다.
- `PixelDataConverter`가 센서 포맷(Mono8/Bayer/YUV 등)을 WPF `Bgra32`에 그대로 맞는 BGRA8로 변환합니다.
- WPF 오브젝트는 UI 스레드에서만 만져야 하므로 `Dispatcher.Invoke`로 마샬링합니다.

## 학습 데이터 수집 (크롭 + 레코딩)

**REC는 크롭 전 풀사이즈 원본을 저장합니다.** 크롭은 세그멘테이션 입력에만 실제로 걸리고,
화면에는 네모로 표시만 됩니다. 학습용 크롭은 `finetuner/preprocess.py` 가 담당합니다 —
이미지만이 아니라 폴리곤 라벨까지 같이 잘라야 하고, 여기서 버린 픽셀은 되돌릴 수 없기 때문입니다.

- **툴바 2번째 줄**: `Crop` 체크 · 크기(px) · `↔`/`↕` 슬라이더(0~100%) · `Center`(둘 다 50%로 리셋)
  - 0% = 왼쪽/위 끝, 100% = 오른쪽/아래 끝, 50% = 중앙.
  - 조절하면 영상 위 빨간 네모가 실시간으로 따라 움직이고, 네모 위에 `640×640 (50%, 50%)`
    처럼 현재 값이 표시됩니다.
  - 크롭 크기가 센서 이미지보다 크면 원본 크기로 클램프되고, 네모는 사라집니다(= 잘릴 게 없음).
  - 여기서 눈으로 찾은 위치(%)를 `preprocess_config.yaml` 의 `crop.center_x/center_y` 에
    그대로 옮겨 적으면 됩니다. 두 곳이 같은 0~100% 규약을 씁니다.
- **`● REC` 버튼**: 한 번 누르면 저장 시작, 다시 누르면 중지. 저장 현황은 상태바 오른쪽에 표시.
  - 저장 속도는 `<Recording><Fps>` (기본 `2`). 연속 프레임은 거의 똑같아서 전부 저장하면
    디스크와 라벨링 시간만 낭비됩니다. 그랩은 `<FrameRate>` 그대로 돌고 저장만 샘플링합니다.
    `0` 으로 두면 모든 프레임을 저장합니다.
  - PNG 인코딩은 별도 워커 스레드 + 바운디드 큐라서 그랩 루프와 UI를 막지 않습니다.
    디스크가 못 따라가면 최신 프레임을 버리고(`dropped`), 파일 번호에 구멍으로 남습니다.
  - 저장되는 건 **카메라 원본**입니다 (세그멘테이션 오버레이가 찍히지 않음).
- 기본 저장 경로 `datasets/raw/images/train`. 라벨을 `datasets/raw/labels/train` 에 넣고
  `datasets/raw/data.yaml` 을 만들면 `preprocess.py` 의 `raw` 소스로 그대로 읽힙니다:

  ```yaml
  names: {0: wire}
  train: images/train
  ```

- 크기·위치 시작값과 저장 경로·저장 fps 는 `config\config.xml` 의 `<Crop>` / `<Recording>` 에서 바꿉니다.
  상대경로는 저장소 루트(= `.git` 이 있는 폴더) 기준입니다.

## 실행

```powershell
dotnet run --project src\BaslerLiveView          # 실제 카메라
dotnet run --project src\BaslerLiveView -- --emulate   # 카메라 없이 에뮬레이터로 테스트
```

또는 빌드 후 `bin\Debug\net8.0-windows\BaslerLiveView.exe` 직접 실행.

## 참고

- pylon이 시스템에 설치돼 있지 않아, 네이티브 런타임(`PylonBase`, `PylonC`, 전송 계층 `*_TL.dll`)을
  `pylon\Runtime\x64`에서 실행 파일 옆으로 복사합니다. 앱 실행 폴더는 항상 네이티브 DLL 검색 경로에 포함됩니다.
- `Basler.Pylon.dll`은 x64 전용이므로 프로세스도 x64로 강제(`PlatformTarget=x64`)합니다.
git submodule update --init --recursive