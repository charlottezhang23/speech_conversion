# 生成用于功能验证的中文双人访谈测试音频（Windows SAPI 合成语音）
# 说明：合成语音只用于验证流水线是否跑通，不能代表真实录音的识别准确率。
param(
    [string]$OutDir = "C:\src\speech_conversion\tests\fixtures\audio"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$lines = @(
    @{ speaker = "A"; text = "你好，谢谢你今天接受采访。先请你简单介绍一下自己。" },
    @{ speaker = "B"; text = "好的。我今年二十九岁，在杭州做产品设计，家里养了两只猫。" },
    @{ speaker = "A"; text = "两只猫分别是什么品种？" },
    @{ speaker = "B"; text = "一只是英国短毛猫，另一只是挪威森林猫，都是三年前开始养的。" },
    @{ speaker = "A"; text = "平时养猫最让你困扰的事情是什么？" },
    @{ speaker = "B"; text = "最麻烦的是掉毛。挪威森林猫毛特别长，每天都要吸地，不然沙发上全是毛。" },
    @{ speaker = "A"; text = "那你会用什么工具来处理？" },
    @{ speaker = "B"; text = "我用过三款吸尘器，现在用的是一款手持的，但续航只有二十分钟，打扫一半就没电了。" },
    @{ speaker = "A"; text = "你买吸尘器的时候主要看什么？" },
    @{ speaker = "B"; text = "我最看重吸力和续航，价格的话两千块以内可以接受，超过三千我就要再想想。" },
    @{ speaker = "A"; text = "如果有一个新产品能解决续航问题，你会考虑吗？" },
    @{ speaker = "B"; text = "会考虑，但是我需要先看到真实的测评，不太相信广告。" },
    @{ speaker = "A"; text = "好的，最后一个问题，你平时会记录养猫的日常吗？" },
    @{ speaker = "B"; text = "偶尔会拍照发在社交平台上，但不会写很长的文字。" }
)

$voice = New-Object -ComObject SAPI.SpVoice
$voices = $voice.GetVoices()
$zhIndex = -1
for ($i = 0; $i -lt $voices.Count; $i++) {
    if ($voices.Item($i).GetDescription() -like "*Chinese*") { $zhIndex = $i; break }
}
if ($zhIndex -lt 0) {
    Write-Error "未找到中文语音（Microsoft Huihui）。无法生成中文测试音频。"
    exit 1
}
$voice.Voice = $voices.Item($zhIndex)
Write-Host "使用语音：$($voices.Item($zhIndex).GetDescription())"

$parts = @()
$i = 0
foreach ($line in $lines) {
    $i++
    # 两位说话人用不同音高模拟，便于人工分辨
    $pitch = if ($line.speaker -eq "A") { "+8%" } else { "-12%" }
    $rate = if ($line.speaker -eq "A") { "0" } else { "-1" }
    $xml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'><prosody pitch='$pitch' rate='$rate'>$($line.text)</prosody></speak>"
    $path = Join-Path $OutDir ("line_{0:00}_{1}.wav" -f $i, $line.speaker)
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Format.Type = 22   # SAFT22kHz16BitMono
    $stream.Open($path, 3)
    $old = $voice.AudioOutputStream
    $voice.AudioOutputStream = $stream
    $voice.Speak($xml, 8) | Out-Null
    $stream.Close()
    $voice.AudioOutputStream = $old
    $parts += $path
}

# 生成 0.6s 静音
$silence = Join-Path $OutDir "silence.wav"
$root = Split-Path -Parent $PSScriptRoot
$ffmpeg = Join-Path $root "tools\ffmpeg\bin\ffmpeg.exe"
if (-not (Test-Path $ffmpeg)) { $ffmpeg = "ffmpeg" }
& $ffmpeg -y -hide_banner -loglevel error -f lavfi -i anullsrc=r=22050:cl=mono -t 0.6 -c:a pcm_s16le $silence

$list = Join-Path $OutDir "concat.txt"
"file '$($silence -replace '\\','/')'" | Set-Content -Path $list -Encoding ASCII
foreach ($p in $parts) {
    "file '$($p -replace '\\','/')'" | Add-Content -Path $list -Encoding ASCII
    "file '$($silence -replace '\\','/')'" | Add-Content -Path $list -Encoding ASCII
}

$out = Join-Path $OutDir "zh_interview_2speakers.wav"
& $ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i $list -ac 1 -ar 16000 -c:a pcm_s16le $out
Remove-Item $list -Force
$parts | ForEach-Object { Remove-Item $_ -Force }
Remove-Item $silence -Force

Write-Host "已生成：$out"
& $ffmpeg -hide_banner -i $out 2>&1 | Select-String "Duration"
