# BaslerLiveView

Basler pylon 카메라를 연결해 실시간 영상을 WPF 창에 띄우는 최소 예제.

## 구성

| 파일 | 역할 |
|------|------|
| `CameraService.cs` | pylon `Camera` 래퍼 — 연결/연속 그랩/`PixelDataConverter`(→BGRA) |
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
      → Dispatcher.Invoke → WriteableBitmap.WritePixels  // UI 표시
```

- 그랩은 pylon이 제공하는 별도 스레드에서 돌고, 프레임마다 `ImageGrabbed`가 발생합니다.
- `PixelDataConverter`가 센서 포맷(Mono8/Bayer/YUV 등)을 WPF `Bgra32`에 그대로 맞는 BGRA8로 변환합니다.
- WPF 오브젝트는 UI 스레드에서만 만져야 하므로 `Dispatcher.Invoke`로 마샬링합니다.

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
