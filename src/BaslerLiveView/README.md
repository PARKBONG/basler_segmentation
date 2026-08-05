# BaslerLiveView

Basler pylon 카메라 + YOLO 세그멘테이션을 하나의 pull API 뒤에 두고, 실시간 모니터링과
CTWD(contact tip to work distance) 측정에 쓰는 앱.

## 구성

| 파일 | 역할 |
|------|------|
| `VisionCam.cs` | 카메라 + 추론. 외부 API는 `connect()` / `get()` / `disconnect()`. `Frame`·`Instance`·`BBox`·`Measurement` 데이터 계약도 이 파일에 있음 |
| `MainWindow.xaml(.cs)` | 펌프 루프 1개 + Gray8 표시 + 벡터 오버레이 |
| `App.xaml.cs` | `--emulate` 인자로 소프트웨어 카메라 에뮬레이터 활성화 |
| `BaslerLiveView.csproj` | `Basler.Pylon.dll` 참조 + `pylon\Runtime\x64` 네이티브 DLL 복사 |

설정 파일도, UI 설정 컨트롤도 없습니다. 촬영 설정은 `VisionCam` 생성자 인자이고,
앱이 쓰는 값은 `MainWindow.xaml.cs` 상단 상수에 있습니다.

## 동작 흐름

```
VisionCam.connect()
  → Open + ROI(640×640, 위치 0-100%) + 픽셀 포맷 + 노출/fps + YOLO 세션
  → StreamGrabber.Start(LatestImages, ProvidedByUser)   // 출력 큐 1개 = 최신만 유지

펌프 루프 (앱 소유, 백그라운드 태스크 1개)
  var f = cam.get();              // 블로킹. null = 신호 없음
      → Dispatcher.Invoke(Render) // 표시 (UI 스레드)
```

- **Pull 모델**입니다. 카메라가 콜백을 부르지 않으므로 콜백 스레드 규약도, 공유 버퍼 보호도,
  종료 레이스도 없습니다. 소비자가 원할 때 한 장씩 가져갑니다.
- **최신 프레임만 유지**합니다(`OutputQueueSize=1`). 소비자가 느리면 그 사이 프레임은 버려지고
  `Frame.Seq` 가 건너뜁니다 — 실시간 모니터링에서 의도된 동작입니다.
- **추론은 `get()` 안에서 인라인**으로 돕니다. 픽셀과 검출이 같은 `Frame` 에 담기므로 서로
  어긋날 수 없고, 아무도 읽지 않는 프레임에 GPU를 쓰지 않습니다.
- **ROI 기본값 640×640** 은 세그 모델 입력과 같습니다. 리사이즈가 일어나지 않으므로 bbox 좌표가
  센서 픽셀과 1:1 이고 되돌릴 스케일이 없습니다. ROI 를 넓히면 리사이즈가 다시 생기고
  픽셀→mm 캘리브레이션도 다시 잡아야 합니다. `0` 을 주면 센서 전체를 씁니다.
- **크롭 위치는 `roiXPercent` / `roiYPercent` (0–100)** 로 정합니다. `0` = 좌/상단 끝,
  `50`(기본) = 중앙, `100` = 우/하단 끝. 센서에서 창이 움직일 수 있는 전체 구간
  (`OffsetX/Y` 의 max = 센서 − ROI) 을 백분율로 나눈 위치이며, 카메라 increment 에 맞춰
  내림 정렬합니다. 센서 전체를 쓸 때는 움직일 여지가 없으므로 무시됩니다.
- **픽셀 포맷 기본값은 카메라(pylon) 기본값** 입니다. `pixelFormat: null`(기본) 이면 노드를
  건드리지 않고, `"Mono8"`·`"BayerRG8"` 처럼 문자열을 주면 그 포맷을 요청합니다. 어떤 포맷이든
  `Frame.Gray` 로 나갈 때 Mono8 로 변환되므로 이 값은 링크 대역폭만 바꾸고 데이터 계약은
  그대로입니다. 카메라가 거부하는 포맷은 무시되고 기본값이 유지됩니다.

## 사용

```csharp
// roiWidth/roiHeight 기본 640, 크롭 위치 기본 중앙(50%), 픽셀 포맷은 카메라 기본값
using var cam = new VisionCam(fps: 20, segmentation: true,
                              roiXPercent: 50, roiYPercent: 50, pixelFormat: null);
cam.connect();                       // 시리얼을 주면 특정 카메라 선택

while (running)
{
    var f = cam.get(timeoutMs: 500);
    if (f is null) { /* 신호 없음 */ continue; }

    // f.Gray  : byte[H, W]  — WriteableBitmap(Gray8).WritePixels 에 그대로 투입 가능
    // f.Seq   : 프레임 번호. 점프 = 드롭
    // f.TimestampSec : 카메라 클럭. 간격 계산은 반드시 이 값으로
    // f.Instances : bbox + 클래스 + 신뢰도
}
```

프레임이 불규칙하게 버려지므로 시간 미분·필터는 프레임 수가 아니라 `TimestampSec` 기준으로
계산해야 합니다.

## 실행

```powershell
dotnet run --project src\BaslerLiveView          # 실제 카메라
dotnet run --project src\BaslerLiveView -- --emulate   # 카메라 없이 에뮬레이터로 테스트
```

또는 빌드 후 `bin\Debug\net8.0-windows\BaslerLiveView.exe` 직접 실행.
세그멘테이션에는 `Models\yolo26s-seg.onnx` 가 실행 파일 옆에 있어야 합니다(저장소 미포함).

## 참고

- pylon이 시스템에 설치돼 있지 않아, 네이티브 런타임(`PylonBase`, `PylonC`, 전송 계층 `*_TL.dll`)을
  `pylon\Runtime\x64`에서 실행 파일 옆으로 복사합니다. 앱 실행 폴더는 항상 네이티브 DLL 검색 경로에 포함됩니다.
- `Basler.Pylon.dll`은 x64 전용이므로 프로세스도 x64로 강제(`PlatformTarget=x64`)합니다.
- 서브모듈: `git submodule update --init --recursive`
