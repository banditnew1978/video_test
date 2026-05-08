"""
Video Dublaj Pipeline
=====================
Demucs → Whisper → Çeviri → OmniVoice TTS (TTS-Geri Bildirimli Süre Döngüsü) → FFmpeg Birleştirme

Kullanım:
    python pipeline.py --video video.mp4 \
                       --target_language Turkish \
                       --target_language_id tr \
                       --lm_studio_url http://localhost:11434/v1 \
                       --translation_api deepl \
                       --translation_api_key YOUR_KEY
"""

import argparse
import gc
import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

def q(path) -> str:
    """Helper to quote shell arguments."""
    return shlex.quote(str(path))

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("dubbing")

# Ortak Python executable yolu (kullanıcı ortamı)
PYTHON_BIN = "/usr/local/envs/dubbing/bin/python"


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def clear_gpu_cache():
    """GPU hafızasını temizle."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("GPU önbelleği temizlendi.")

def run(cmd: str, desc: str = "") -> subprocess.CompletedProcess:
    """Shell komutu çalıştır, hata durumunda exception fırlat."""
    log.debug(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Komut başarısız ({desc}):\n{cmd}\nSTDERR: {result.stderr[:500]}"
        )
    return result


def get_audio_duration(path: str) -> float:
    """ffprobe ile ses dosyasının süresini saniye olarak döndür."""
    result = run(
        f'ffprobe -v quiet -show_entries format=duration '
        f'-of csv=p=0 {q(path)}',
        "ffprobe duration"
    )
    return float(result.stdout.strip())


def get_video_duration(path: str) -> float:
    """Video süresini saniye olarak döndür."""
    return get_audio_duration(path)


def seconds_to_srt_time(secs: float) -> str:
    """12.345 → 00:00:12,345"""
    total_ms = int(round(secs * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list, field: str, out_path: str):
    """Segment listesinden SRT dosyası yaz. field: 'original_text' veya 'final_text'"""
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.get(field, "") or ""
        if not text.strip():
            continue
        lines.append(str(i))
        lines.append(
            f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}"
        )
        lines.append(text.strip())
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    log.info(f"SRT kaydedildi: {out_path}")


def save_segments(segments: list, path: str):
    Path(path).write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_leading_silence(audio_path: str, threshold_db: str = "-40dB") -> float:
    """
    Ses dosyasındaki baştaki sessizlik süresini saniye olarak döndürür.
    ffmpeg silencedetect filtresini kullanır.
    """
    try:
        result = subprocess.run(
            f'ffmpeg -i {q(audio_path)} -af "silencedetect=noise={threshold_db}:d=0.01" '
            f'-f null -',
            shell=True, capture_output=True, text=True
        )
        output = result.stderr
        for line in output.split("\n"):
            if "silence_end" in line:
                parts = line.split("silence_end:")[1].split("|")[0].strip()
                return float(parts)
    except Exception:
        pass
    return 0.0


def verify_no_leading_silence(audio_path: str, threshold_db: str = "-40dB", max_allowed_ms: float = 20.0) -> bool:
    """
    TTS çıktısının başında boşluk olmadığını doğrular.
    max_allowed_ms'den fazla sessizlik varsa False döner.
    """
    silence_sec = detect_leading_silence(audio_path, threshold_db)
    silence_ms = silence_sec * 1000
    if silence_ms > max_allowed_ms:
        log.warning(f"Baştaki sessizlik: {silence_ms:.1f}ms > {max_allowed_ms}ms — {audio_path}")
        return False
    return True


def detect_trailing_silence(audio_path: str, threshold_db: str = "-40dB") -> float:
    """
    Ses dosyasındaki sondaki sessizlik süresini saniye olarak döndürür.
    Dosyayı ters çevirip baştaki sessizliği ölçer.
    """
    try:
        # Dosyayı ters çevir ve baştaki sessizliği ölç
        result = subprocess.run(
            f'ffmpeg -i {q(audio_path)} -af "areverse,silencedetect=noise={threshold_db}:d=0.01" '
            f'-f null -',
            shell=True, capture_output=True, text=True
        )
        output = result.stderr
        for line in output.split("\n"):
            if "silence_end" in line:
                parts = line.split("silence_end:")[1].split("|")[0].strip()
                return float(parts)
    except Exception:
        pass
    return 0.0


def trim_audio_silence(input_path: str, output_path: str, threshold_db: str = "-40dB") -> str:
    """
    Ses dosyasının başındaki ve sonundaki sessizliği kırpar.
    Kırpılmış dosya yolunu döndürür.
    """
    trim_filter = (
        f"silenceremove=start_periods=1:start_threshold={threshold_db}:start_duration=0.02,"
        f"areverse,"
        f"silenceremove=start_periods=1:start_threshold={threshold_db}:start_duration=0.02,"
        f"areverse"
    )
    run(
        f'ffmpeg -y -i {q(input_path)} -af "{trim_filter}" {q(output_path)}',
        "trim silence"
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — SES AYIRMA
# ══════════════════════════════════════════════════════════════════════════════

def step1_separate_audio(video_path: str, tmp: str) -> tuple[str, str]:
    """
    Returns: (vocals_path, background_path)
    """
    log.info("=" * 60)
    log.info("ADIM 1 — Ses ayırma (Demucs)")

    raw_audio = f"{tmp}/raw_audio.wav"
    run(
        f'ffmpeg -y -i {q(video_path)} -vn -acodec pcm_s16le -ar 44100 {q(raw_audio)}',
        "video→wav"
    )
    log.info(f"Ham ses çıkarıldı: {raw_audio}")

    # Demucs ile kaynak ayrıştırma
    run(
        f'{q(PYTHON_BIN)} -m demucs --two-stems=vocals -o {q(tmp + "/demucs")} {q(raw_audio)}',
        "demucs"
    )

    # Demucs çıktı dizini bul (model adını otomatik algıla)
    demucs_out = Path(f"{tmp}/demucs")
    model_dirs = list(demucs_out.iterdir())
    if not model_dirs:
        raise RuntimeError("Demucs çıktı dizini bulunamadı!")
    model_dir = model_dirs[0]

    audio_stem = Path(raw_audio).stem
    vocals_src = model_dir / audio_stem / "vocals.wav"
    bg_src     = model_dir / audio_stem / "no_vocals.wav"

    vocals_path = f"{tmp}/vocals.wav"
    bg_path     = f"{tmp}/no_vocals.wav"
    shutil.copy(vocals_src, vocals_path)
    shutil.copy(bg_src, bg_path)

    log.info(f"Vokal: {vocals_path}")
    log.info(f"Arka plan: {bg_path}")
    return vocals_path, bg_path


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — TRANSKRİPSİYON
# ══════════════════════════════════════════════════════════════════════════════

import re

def timestamp_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def parse_diarize_srt(srt_path: str) -> list:
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    pattern = re.compile(
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n\d+\n|$)",
        re.DOTALL,
    )

    segments = []
    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_ts = match.group(2)
        end_ts = match.group(3)
        text_block = match.group(4).strip()
        
        speaker_match = re.match(r"^(Speaker \d+):\s*(.*)", text_block, re.DOTALL)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2).replace("\n", " ").strip()
        else:
            speaker = "Speaker 0"
            text = text_block.replace("\n", " ").strip()

        start = timestamp_to_seconds(start_ts)
        end = timestamp_to_seconds(end_ts)
        
        segments.append({
            "id": idx,
            "speaker": speaker,
            "start": round(start, 3),
            "end":   round(end, 3),
            "duration": round(end - start, 3),
            "original_text": text,
            "ref_audio_path": None,
            "ref_audio_duration": None,
            "translated_text": None,
            "final_text": None,
            "needs_shortening": False,
            "tts_output_path": None,
            "tts_duration": None,
            "speed_adjusted": False,
            "speed_factor": 1.0,
            "warning": None,
        })
    return segments

def step2_diarize(vocals_path: str, whisper_model: str, tmp: str, source_language: str = "auto") -> list:
    """
    Diarize.py ile transkripsiyon ve konuşmacı ayrımı.
    """
    log.info("=" * 60)
    log.info(f"ADIM 2 — Diarization & Transcription (Whisper {whisper_model}, dil={source_language})")

    diarize_script = Path(__file__).resolve().parent / "diarize.py"
    python_bin = PYTHON_BIN
    if not diarize_script.exists():
        raise RuntimeError(f"diarize.py bulunamadı: {diarize_script}")
    
    cmd = f"{q(python_bin)} {q(diarize_script)} -a {q(vocals_path)} --no-stem --whisper-model {whisper_model} --device cuda"
    if source_language != "auto":
        cmd += f" --language {source_language}"
        
    run(cmd, "diarize.py")
    
    import os
    base_name = os.path.splitext(vocals_path)[0]
    srt_out = f"{base_name}.srt"
    
    if not os.path.exists(srt_out):
        raise RuntimeError(f"Diarize çıktısı bulunamadı: {srt_out}")
        
    segments = parse_diarize_srt(srt_out)
    
    seg_path = f"{tmp}/segments.json"
    save_segments(segments, seg_path)
    shutil.copy(srt_out, f"{tmp}/original.srt")
    log.info(f"{len(segments)} segment transkribe edildi ve konuşmacılara ayrıldı.")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2.5 — KONUŞMACI BLOKLAMA (Speaker Merging)
# ══════════════════════════════════════════════════════════════════════════════

def step2_5_merge_speakers(segments: list, gap_threshold_ms: int, tmp: str) -> list:
    """
    Ardışık aynı konuşmacı segmentlerini birleştirir.
    
    Kural: İki ardışık segment aynı konuşmacıya aitse VE aralarındaki
    boşluk gap_threshold_ms'den küçükse, tek bir blok olarak birleştirilir.
    
    Bu sayede OmniVoice TTS'e daha uzun ve doğal metin blokları gönderilir,
    daha akıcı bir seslendirme elde edilir.
    """
    log.info("=" * 60)
    log.info(f"ADIM 2.5 — Konuşmacı Bloklama (eşik: {gap_threshold_ms}ms)")
    
    if not segments:
        return segments
    
    gap_threshold_sec = gap_threshold_ms / 1000.0
    merged = []
    current_block = dict(segments[0])  # İlk segmentle başla (kopya)
    current_block["merged_ids"] = [current_block["id"]]
    
    for i in range(1, len(segments)):
        seg = segments[i]
        gap = seg["start"] - current_block["end"]
        same_speaker = seg["speaker"] == current_block["speaker"]
        
        if same_speaker and gap <= gap_threshold_sec:
            # Birleştir: metni ekle, bitiş zamanını güncelle
            current_block["original_text"] += " " + seg["original_text"]
            current_block["end"] = seg["end"]
            current_block["duration"] = round(current_block["end"] - current_block["start"], 3)
            current_block["merged_ids"].append(seg["id"])
        else:
            # Farklı konuşmacı veya boşluk çok büyük → bloğu kapat, yenisini başlat
            merged.append(current_block)
            current_block = dict(seg)
            current_block["merged_ids"] = [seg["id"]]
    
    # Son bloğu ekle
    merged.append(current_block)
    
    # ID'leri yeniden numaralandır
    for i, block in enumerate(merged, 1):
        block["id"] = i
    
    original_count = len(segments)
    merged_count = len(merged)
    reduced = original_count - merged_count
    
    log.info(f"  Orijinal: {original_count} segment → Birleştirilmiş: {merged_count} blok ({reduced} segment birleştirildi)")
    
    # Birleştirilen blokları logla
    for block in merged:
        if len(block["merged_ids"]) > 1:
            ids_str = ", ".join(str(x) for x in block["merged_ids"])
            log.info(
                f"  Blok {block['id']}: [{block['speaker']}] "
                f"{block['start']:.3f}s → {block['end']:.3f}s "
                f"({block['duration']:.2f}s) — orijinal ID'ler: [{ids_str}]"
            )
    
    save_segments(merged, f"{tmp}/segments.json")
    write_srt(merged, "original_text", f"{tmp}/original_merged.srt")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — REFERANS SES SEGMENTLERINI ÇIKAR
# ══════════════════════════════════════════════════════════════════════════════

def step3_extract_ref_segments(segments: list, vocals_path: str, tmp: str, silence_threshold: str = "-40dB") -> list:
    log.info("=" * 60)
    log.info("ADIM 3 — Referans ses segmentleri (Her konuşmacı için 1 adet) çıkarılıyor")

    total_duration = get_audio_duration(vocals_path)
    
    from collections import defaultdict
    speakers = defaultdict(list)
    for seg in segments:
        speakers[seg["speaker"]].append(seg)
        
    speaker_refs = {}
    
    for spk, spk_segments in speakers.items():
        best_seg = None
        for s in sorted(spk_segments, key=lambda x: x["duration"], reverse=True):
            if 3.0 <= s["duration"] <= 10.0:
                best_seg = s
                break
        if not best_seg:
            best_seg = max(spk_segments, key=lambda x: x["duration"])
            
        start = best_seg["start"]
        end = best_seg["end"]
        dur = best_seg["duration"]
        
        extract_start = start
        extract_end = end
        if dur < 3.0:
            needed = 3.0 - dur
            extra_after = min(needed / 2, total_duration - end)
            extra_before = min(needed - extra_after, start)
            extract_start = max(0.0, start - extra_before)
            extract_end = min(total_duration, end + extra_after)
            
        if (extract_end - extract_start) > 10.0:
            extract_end = extract_start + 10.0
            
        spk_safe = spk.replace(" ", "_")
        raw_out = f"{tmp}/ref_raw_{spk_safe}.wav"
        trimmed_out = f"{tmp}/ref_{spk_safe}.wav"
        
        try:
            run(
                f'ffmpeg -y -i {q(vocals_path)} '
                f'-ss {extract_start:.3f} -to {extract_end:.3f} '
                f'-acodec pcm_s16le -ar 24000 -ac 1 {q(raw_out)}',
                f"ref_extract_{spk_safe}"
            )
            trim_audio_silence(raw_out, trimmed_out, threshold_db=silence_threshold)
            ref_dur = get_audio_duration(trimmed_out)
            speaker_refs[spk] = (trimmed_out, ref_dur)
            log.info(f"{spk} için referans ses oluşturuldu: {trimmed_out} ({ref_dur:.2f}s)")
        except Exception as e:
            log.warning(f"{spk} referans ses çıkarılamadı: {e}")
            speaker_refs[spk] = None

    for seg in segments:
        ref_data = speaker_refs.get(seg["speaker"])
        if ref_data is not None:
            seg["ref_audio_path"] = ref_data[0]
            seg["ref_audio_duration"] = ref_data[1]
        else:
            seg["ref_audio_path"] = None
            seg["ref_audio_duration"] = None
        seg["original_start"] = seg["start"]
        seg["original_end"] = seg["end"]
        seg["original_duration"] = seg["duration"]
        
    save_segments(segments, f"{tmp}/segments.json")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3.5 — SES CİNSİYETİ TESPİTİ VE ÖZEL SES ATAMASI
# ══════════════════════════════════════════════════════════════════════════════

def step3_5_assign_voices(segments: list, tmp: str, no_confirm: bool):
    log.info("=" * 60)
    log.info("ADIM 3.5 — Ses Cinsiyeti Tespiti ve Referans Ses Ataması")
    
    try:
        from voice_gender_classification import GenderClassificationPipeline
    except ImportError:
        log.error("voice-gender-classification kurulu değil. Ses ataması atlanıyor.")
        return segments
        
    voice_dir = Path("voice")
        
    # Kullanılabilir özel sesleri oku (voice/ klasörü)
    available_voices = {"male": [], "female": []}
    if voice_dir.exists():
        for wav_file in voice_dir.glob("*.wav"):
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
                
            with open(txt_file, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
                
            name = wav_file.stem.lower()
            if "male" in name and "female" not in name:
                available_voices["male"].append({"name": wav_file.stem, "wav": str(wav_file), "txt": ref_text})
            elif "female" in name:
                available_voices["female"].append({"name": wav_file.stem, "wav": str(wav_file), "txt": ref_text})
            else:
                available_voices["male"].append({"name": wav_file.stem, "wav": str(wav_file), "txt": ref_text})

    has_custom_voices = bool(available_voices["male"] or available_voices["female"])

    # Cinsiyet modeli yükle
    log.info("Cinsiyet sınıflandırma modeli yükleniyor...")
    try:
        gender_clf = GenderClassificationPipeline.from_pretrained("griko/gender_cls_svm_ecapa_voxceleb")
    except Exception as e:
        log.error(f"Cinsiyet modeli yüklenemedi: {e}")
        return segments

    # Benzersiz konuşmacıları bul
    speakers = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in speakers:
            speakers[spk] = seg.get("ref_audio_path")
            
    speaker_voice_map = {}
    
    for spk, ref_path in speakers.items():
        if not ref_path or not Path(ref_path).exists():
            log.warning(f"{spk} için referans ses bulunamadı, atlanıyor.")
            continue
            
        # Cinsiyet tespiti
        gender = "bilinmiyor"
        try:
            res = gender_clf(ref_path)
            if res and isinstance(res, list):
                gender = res[0].lower()
        except Exception as e:
            log.warning(f"{spk} cinsiyeti tespit edilemedi: {e}")
            
        log.info(f"{spk} cinsiyeti tespit edildi: {gender}")
        
        # Seçenek listesi oluştur
        # [0] = Kendi sesi (videodan çıkarılan referans)
        # [1..N] = voice/ klasöründeki özel sesler
        custom_choices = available_voices.get(gender, [])
        if not custom_choices and has_custom_voices:
            custom_choices = available_voices["male"] + available_voices["female"]

        if no_confirm:
            # --no_confirm modunda kendi sesini kullan (videodan)
            speaker_voice_map[spk] = {"voice": None, "tone_convert": False}
            log.info(f"--no_confirm aktif, {spk} için kendi sesi (videodan) kullanılacak.")
        else:
            print("\n" + "-"*50)
            print(f"🎤 {spk} tespit edildi: [{gender.upper()}]")
            print(f"  [0] 🎬 Kendi sesi (videodan çıkarılan orijinal ses)")
            if custom_choices:
                print("  --- Özel referans sesler (voice/ klasörü) ---")
                for i, v in enumerate(custom_choices):
                    print(f"  [{i+1}] 🎙️  {v['name']}")
            
            max_choice = len(custom_choices)
            selected_voice = None
            while True:
                secim = input(f"{spk} için hangi sesi kullanmak istiyorsunuz? (0-{max_choice}): ").strip()
                if secim.isdigit() and 0 <= int(secim) <= max_choice:
                    secim_int = int(secim)
                    if secim_int == 0:
                        selected_voice = None
                        print(f"  ✅ {spk} → Kendi sesi (videodan)")
                    else:
                        selected_voice = custom_choices[secim_int - 1]
                        print(f"  ✅ {spk} → {selected_voice['name']}")
                    break
                else:
                    print("Geçersiz seçim, tekrar deneyin.")
            
            # Özel ses seçildiyse ton rengi dönüşümü sor
            tone_convert = False
            if selected_voice is not None:
                print(f"  🎨 {spk} için orijinal konuşmacının ton rengini korumak ister misiniz?")
                print(f"     (Özel ses ile seslendirme yapılır, ardından orijinal konuşmacının tonu aktarılır)")
                tc_cevap = input(f"  Ton rengi dönüşümü uygulansın mı? (e/h) [Varsayılan: h]: ").lower().strip()
                if tc_cevap == 'e':
                    tone_convert = True
                    print(f"  ✅ Ton rengi dönüşümü AKTİF — {spk}")
            
            speaker_voice_map[spk] = {"voice": selected_voice, "tone_convert": tone_convert}
        
    # Segmentleri güncelle
    for seg in segments:
        spk = seg["speaker"]
        if spk in speaker_voice_map:
            entry = speaker_voice_map[spk]
            selected = entry["voice"]
            if selected is not None:
                # Özel ses seçildi → voice/ klasöründen
                seg["original_ref_audio_path"] = seg.get("ref_audio_path")  # Orijinal sesi sakla (ton dönüşümü için)
                seg["ref_audio_path"] = selected["wav"]
                seg["ref_text"] = selected["txt"]
                seg["voice_source"] = f"custom:{selected['name']}"
                seg["tone_convert"] = entry["tone_convert"]
            else:
                # Kendi sesi → step3'te oluşturulan referans ses kalacak
                seg["voice_source"] = "original"
                seg["tone_convert"] = False
            
    save_segments(segments, f"{tmp}/segments.json")
    log.info("Ses atamaları tamamlandı.")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — ÇEVİRİ
# ══════════════════════════════════════════════════════════════════════════════

MAX_CHARS_PER_SECOND = 21
MIN_CHARS_PER_SECOND = 14

def _call_lm_studio(url: str, model: str, system_prompt: str, user_prompt: str) -> str:
    import urllib.request
    import urllib.error
    import json
    
    payload_dict = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }
    payload = json.dumps(payload_dict).encode()
    
    # URL düzeltmesi
    url_clean = url.rstrip('/')
    if not url_clean.endswith("/chat/completions"):
        if not url_clean.endswith("/v1"):
            url_clean = f"{url_clean}/v1"
        url_clean = f"{url_clean}/chat/completions"
        
    req = urllib.request.Request(
        url_clean,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp_bytes = r.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"HTTP {e.code}: {error_body}")
        
    try:
        resp = json.loads(resp_bytes)
    except json.JSONDecodeError:
        raise Exception(f"JSON Decode hatası: {resp_bytes}")
        
    if "choices" not in resp:
        raise Exception(f"API Yanıtında 'choices' yok. Gelen Yanıt: {resp}")
        
    content = resp["choices"][0]["message"]["content"].strip()
    return content

def step4_translate(
    segments: list,
    target_language: str,
    lm_studio_url: str,
    lm_studio_model: str,
    tmp: str,
    translation_strategy: str = "compress",
    chars_per_second: float = 21.0,
    context_size: int = 2,
) -> list:
    log.info("=" * 60)
    log.info(f"ADIM 4 — Bağlamsal Çeviri (Strateji: {translation_strategy}, Bağlam: {context_size}) (Ollama/OpenAI API -> {target_language})")

    SYSTEM_TRANSLATE_CONTEXT = f"""
You are a professional subtitle localizer.
Translate subtitles into natural spoken {target_language}.

Context rules:
- I will provide Previous context, the Current subtitle, and Next context.
- Translate ONLY the Current subtitle text.
- Previous and Next are ONLY for understanding flow - DO NOT include any part of them in your translation.
- Your output must match ONLY the Current subtitle content.

General rules:
- Your translation MUST be between {{min_chars}} and {{max_chars}} characters.
- Aim to use as much of the allowed character budget as possible to produce natural-sounding speech.
- Do NOT over-shorten. Fill the available time with natural, expressive {target_language}.
- Preserve meaning, not literal words.
- Return ONLY final {target_language} text, no explanations.

TTS Pronunciation rules (CRITICAL):
- KEEP all numbers as digits (e.g. "$106" → "106 dolar", "30%" → "%30"). Do NOT convert numbers to words, this will be done automatically later.
- Transliterate foreign brand names and English words into {target_language} phonetic spelling so a TTS engine reads them correctly (e.g. "Intel" → "İntel", "Nasdaq" → "Nasdak", "YouTube" → "Yutub", "iPhone" → "Ayfon").
- Do NOT use abbreviations. Everything must be readable as spoken {target_language}.
""".strip()

    SYSTEM_TRANSLATE_CONTEXT_NO_LIMIT = f"""
You are a professional subtitle localizer.
Translate subtitles into natural spoken {target_language}.

Context rules:
- I will provide Previous context, the Current subtitle, and Next context.
- Translate ONLY the Current subtitle text.
- Previous and Next are ONLY for understanding flow - DO NOT include any part of them in your translation.
- Your output must match ONLY the Current subtitle content.

General rules:
- Preserve meaning, not literal words.
- Keep {target_language} concise and natural.
- Return ONLY final {target_language} text, no explanations.

TTS Pronunciation rules (CRITICAL):
- KEEP all numbers as digits. Do NOT convert numbers to words.
- Transliterate foreign brand names and English words into {target_language} phonetic spelling (e.g. "Intel" → "İntel", "Nasdaq" → "Nasdak", "YouTube" → "Yutub").
- Do NOT use abbreviations.
""".strip()

    SYSTEM_COMPRESS = f"""
You are a subtitle compression expert.

Slightly shorten {target_language} subtitles to fit the speaking duration.

Rules:
- Preserve original meaning as much as possible
- Only remove unnecessary words, do NOT over-compress
- The result MUST be between {{min_chars}} and {{max_chars}} characters
- Keep the translation natural and expressive — do NOT make it telegraphic
- Natural spoken {target_language} only
- No explanations
- Return ONLY final compressed subtitle
""".strip()

    for i, seg in enumerate(segments):
        # Bağlam penceresi: önceki ve sonraki N segment
        prev_texts = []
        for j in range(max(0, i - context_size), i):
            prev_texts.append(segments[j]["original_text"])
        
        next_texts = []
        for j in range(i + 1, min(len(segments), i + 1 + context_size)):
            next_texts.append(segments[j]["original_text"])
        
        prev_context = " | ".join(prev_texts) if prev_texts else "(yok)"
        next_context = " | ".join(next_texts) if next_texts else "(yok)"
        current_text = seg["original_text"]
        dur = seg["duration"]
        
        cps_max = chars_per_second
        cps_min = max(10.0, chars_per_second * 0.67)
        max_chars = int(dur * cps_max)
        min_chars = int(dur * cps_min)
        
        if translation_strategy == "speed_up":
            system_prompt = SYSTEM_TRANSLATE_CONTEXT_NO_LIMIT
            user_prompt = f"Duration: {dur:.2f}s\n\nPrevious: {prev_context}\nCurrent: {current_text}\nNext: {next_context}"
        else:
            system_prompt = SYSTEM_TRANSLATE_CONTEXT.format(min_chars=min_chars, max_chars=max_chars)
            user_prompt = f"Duration: {dur:.2f}s (Target: {min_chars}-{max_chars} chars)\n\nPrevious: {prev_context}\nCurrent: {current_text}\nNext: {next_context}"
            
        log.info(f"Çevriliyor [{i+1}/{len(segments)}]: {current_text}")
        try:
            translated = _call_lm_studio(lm_studio_url, lm_studio_model, system_prompt, user_prompt)
            
            if translation_strategy == "compress" and len(translated) > max_chars:
                log.info(f"  (!) Çok uzun ({len(translated)} > {max_chars}), sıkıştırılıyor...")
                compress_prompt = SYSTEM_COMPRESS.format(min_chars=min_chars, max_chars=max_chars)
                comp_user_prompt = f"Target duration: {dur:.2f} seconds (Target: {min_chars}-{max_chars} chars)\n\nSubtitle:\n{translated}"
                translated = _call_lm_studio(lm_studio_url, lm_studio_model, compress_prompt, comp_user_prompt)
                
            seg["translated_text"] = translated
            seg["final_text"] = translated
            
        except Exception as e:
            log.error(f"Çeviri hatası (Segment {seg['id']}): {e}")
            seg["translated_text"] = current_text
            seg["final_text"] = current_text
            seg["warning"] = f"Çeviri hatası: {e}"

    save_segments(segments, f"{tmp}/segments.json")
    write_srt(segments, "translated_text", f"{tmp}/translated_raw.srt")
    log.info("Çeviri ve sıkıştırma tamamlandı.")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4.5 — TTS Metin Normalizasyonu (Rakamlar → Yazı, Fonetik Düzeltme)
# ══════════════════════════════════════════════════════════════════════════════

def _add_spaces_to_turkish_number(word: str) -> str:
    """
    num2words Türkçe çıktısına okunabilirlik için boşluk ekler.
    Örn: 'beşyüz' → 'beş yüz', 'ikibinyirmidört' → 'iki bin yirmi dört'
    """
    # Türkçe sayı kelimeleri (uzundan kısaya sıralı — greedy match)
    tokens = [
        'milyar', 'milyon', 'trilyon',
        'yüz', 'bin',
        'altmış', 'yetmiş', 'seksen', 'doksan',
        'yirmi', 'otuz', 'kırk', 'elli',
        'virgül', 'nokta',
        'on',
        'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz',
        'sıfır',
    ]
    
    result = []
    remaining = word.lower().strip()
    
    while remaining:
        matched = False
        for token in tokens:
            if remaining.startswith(token):
                result.append(token)
                remaining = remaining[len(token):]
                matched = True
                break
        if not matched:
            # Bilinmeyen karakter, olduğu gibi ekle
            result.append(remaining[0])
            remaining = remaining[1:]
    
    return ' '.join(result)


def normalize_text_for_tts(text: str, lang: str = "tr") -> str:
    """
    TTS motoru için metni normalize eder:
    - Rakamları yazıyla yazar (num2words)
    - Yüzde/para sembollerini açar
    """
    import re
    
    try:
        from num2words import num2words
    except ImportError:
        log.warning("num2words kurulu değil, rakam dönüştürme atlanıyor.")
        return text
    
    # Yüzde işareti: %30 veya 30% → yüzde otuz
    def replace_percent(m):
        num_str = m.group(1) or m.group(2)
        try:
            word = num2words(float(num_str) if '.' in num_str or ',' in num_str else int(num_str), lang=lang)
            return f"yüzde {_add_spaces_to_turkish_number(word)}"
        except Exception:
            return m.group(0)
    
    text = re.sub(r'%\s*([\d.,]+)', replace_percent, text)
    text = re.sub(r'([\d.,]+)\s*%', lambda m: replace_percent(type('M', (), {'group': lambda s, i: m.group(1) if i in (1,2) else m.group(0)})()), text)
    
    # Para birimleri
    currency_map = {
        '$': 'dolar', '€': 'euro', '£': 'sterlin', '₺': 'lira', '¥': 'yen',
    }
    for symbol, name in currency_map.items():
        pattern = re.escape(symbol) + r'\s*([\d.,]+)'
        def make_currency_replacer(curr_name):
            def replacer(m):
                try:
                    num_str = m.group(1).replace(',', '')
                    word = num2words(float(num_str) if '.' in num_str else int(num_str), lang=lang)
                    return f"{_add_spaces_to_turkish_number(word)} {curr_name}"
                except Exception:
                    return m.group(0)
            return replacer
        text = re.sub(pattern, make_currency_replacer(name), text)
        pattern2 = r'([\d.,]+)\s*' + re.escape(symbol)
        text = re.sub(pattern2, make_currency_replacer(name), text)
    
    # ── 1) Binlik ayırıcılı sayılar (1.500.000, 1,000,000) — EN ÖNCE ──
    def replace_thousands(m):
        num_str = m.group(0).replace('.', '').replace(',', '')
        try:
            word = num2words(int(num_str), lang=lang)
            return _add_spaces_to_turkish_number(word)
        except Exception:
            return m.group(0)
    
    text = re.sub(r'\d{1,3}(?:[.,]\d{3})+', replace_thousands, text)
    
    # ── 2) Ondalık sayılar (3.14, 2,5 vb.) ──
    def replace_decimal(m):
        num_str = m.group(0)
        try:
            if ',' in num_str and '.' not in num_str:
                num_val = float(num_str.replace(',', '.'))
            else:
                num_val = float(num_str)
            word = num2words(num_val, lang=lang)
            return _add_spaces_to_turkish_number(word)
        except Exception:
            return num_str
    
    text = re.sub(r'\d+[.,]\d+', replace_decimal, text)
    
    # ── 3) Tam sayılar (kalan) ──
    def replace_integer(m):
        try:
            word = num2words(int(m.group(0)), lang=lang)
            return _add_spaces_to_turkish_number(word)
        except Exception:
            return m.group(0)
    
    text = re.sub(r'\b\d+\b', replace_integer, text)
    
    # Çoklu boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def step4_5_normalize_for_tts(segments: list, target_language_id: str, tmp: str) -> list:
    """
    Çevrilmiş metinleri TTS için normalize eder.
    Rakamları yazıya çevirir, sembolleri açar.
    """
    log.info("=" * 60)
    log.info("ADIM 4.5 — TTS Metin Normalizasyonu")
    
    changed_count = 0
    for seg in segments:
        original = seg.get("final_text", "")
        if not original:
            continue
        
        normalized = normalize_text_for_tts(original, lang=target_language_id)
        
        if normalized != original:
            log.info(f"  Seg {seg['id']}: '{original}' → '{normalized}'")
            seg["final_text"] = normalized
            seg["pre_tts_normalized"] = True
            changed_count += 1
    
    log.info(f"Normalizasyon tamamlandı: {changed_count}/{len(segments)} segment dönüştürüldü.")
    save_segments(segments, f"{tmp}/segments.json")
    write_srt(segments, "final_text", f"{tmp}/translated_normalized.srt")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — KALDIRILDI (TTS-geri bildirimli süre döngüsü Adım 6'ya taşındı)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 6.5 — OpenVoice Ton Rengi Dönüşümü
# ══════════════════════════════════════════════════════════════════════════════

def step6_5_tone_convert(
    segments: list,
    tmp: str,
    device: str = "cuda:0",
) -> list:
    """
    Özel ses ile seslendirilen segmentlerde, orijinal konuşmacının ton rengini
    TTS çıktısına aktarır.
    
    Akış: TTS çıktısı (özel ses tonu) + Orijinal konuşmacı tonu → Ton dönüştürülmüş ses
    """
    # Ton dönüşümü gereken segment var mı kontrol et
    segments_to_convert = [
        s for s in segments
        if s.get("tone_convert") and s.get("tts_output_path") and s.get("original_ref_audio_path")
    ]
    
    if not segments_to_convert:
        log.info("Ton rengi dönüşümü gereken segment yok, atlanıyor.")
        return segments
    
    log.info("=" * 60)
    log.info(f"ADIM 6.5 — OpenVoice Ton Rengi Dönüşümü ({len(segments_to_convert)} segment)")
    
    try:
        import torch
        from openvoice_cli.api import ToneColorConverter
        from openvoice_cli.downloader import download_checkpoint
    except ImportError as e:
        log.error(f"OpenVoice kurulu değil, ton dönüşümü atlanıyor: {e}")
        return segments
    
    # Checkpoint'leri indir (yoksa)
    ckpt_dir = Path(tmp) / "openvoice_ckpt"
    try:
        download_checkpoint(str(ckpt_dir))
    except Exception as e:
        log.error(f"OpenVoice checkpoint indirilemedi: {e}")
        return segments
    
    # Modeli yükle
    config_path = str(ckpt_dir / "config.json")
    ckpt_path = str(ckpt_dir / "checkpoint.pth")
    
    try:
        converter = ToneColorConverter(config_path, device=device)
        converter.load_ckpt(ckpt_path)
        log.info("OpenVoice ToneColorConverter yüklendi.")
    except Exception as e:
        log.error(f"OpenVoice modeli yüklenemedi: {e}")
        return segments
    
    # Konuşmacı bazında orijinal ton embedding'lerini çıkar (bir kez)
    speaker_tone_cache = {}
    MIN_TONE_REF_DURATION = 2.0   # Ton embedding için minimum referans süresi (saniye)
    MIN_TTS_DURATION_FOR_TC = 1.0 # Ton dönüşümü için minimum TTS segment süresi (saniye)
    
    for seg in segments_to_convert:
        sid = seg["id"]
        spk = seg["speaker"]
        tts_path = seg["tts_output_path"]
        orig_ref = seg["original_ref_audio_path"]
        
        try:
            # TTS çıktısı çok kısaysa ton dönüşümünü atla
            tts_dur = seg.get("tts_duration", 0)
            if tts_dur < MIN_TTS_DURATION_FOR_TC:
                log.warning(
                    f"Seg {sid}: TTS süresi çok kısa ({tts_dur:.2f}s < {MIN_TTS_DURATION_FOR_TC}s), "
                    f"ton dönüşümü atlanıyor."
                )
                continue
            
            # Orijinal konuşmacı ton embedding'i (hedef ton)
            if spk not in speaker_tone_cache:
                ref_dur = get_audio_duration(orig_ref)
                if ref_dur < MIN_TONE_REF_DURATION:
                    log.warning(
                        f"{spk}: Referans ses çok kısa ({ref_dur:.2f}s < {MIN_TONE_REF_DURATION}s), "
                        f"ton embedding güvenilir olmayabilir."
                    )
                tgt_se = converter.extract_se([orig_ref])
                speaker_tone_cache[spk] = tgt_se
                log.info(f"{spk}: Orijinal ton embedding'i çıkarıldı ({ref_dur:.2f}s referans).")
            else:
                tgt_se = speaker_tone_cache[spk]
            
            # TTS çıktısından kaynak ton embedding'i
            src_se = converter.extract_se([tts_path])
            
            # Ton dönüşümü uygula
            tc_output = f"{tmp}/tts_{sid}_tc.wav"
            converter.convert(
                audio_src_path=tts_path,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=tc_output,
                tau=0.3,
            )
            
            # Çıktıyı güncelle
            seg["tts_output_path"] = tc_output
            seg["tts_duration"] = round(get_audio_duration(tc_output), 3)
            log.info(f"Seg {sid}: Ton dönüşümü tamamlandı → {tc_output}")
            
        except Exception as e:
            log.warning(f"Seg {sid}: Ton dönüşümü başarısız — {e}. Orijinal TTS çıktısı korunuyor.")
            seg["warning"] = (seg.get("warning") or "") + f" | Ton dönüşümü hatası: {e}"
    
    # Modeli temizle
    del converter
    if 'torch' in dir() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    save_segments(segments, f"{tmp}/segments.json")
    log.info("Ton rengi dönüşümü tamamlandı.")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 6 — OmniVoice TTS + TTS-Geri Bildirimli Süre Döngüsü
# ══════════════════════════════════════════════════════════════════════════════

def step6_tts(
    segments: list,
    omnivoice_model: str,
    omnivoice_device: str,
    target_language_id: str,
    tmp: str,
    silence_threshold: str = "-40dB",
    apply_extra_trim: bool = True,
):
    log.info("=" * 60)
    log.info("ADIM 6 — OmniVoice TTS")

    try:
        import torch
        import soundfile as sf
        from omnivoice import OmniVoice
    except ImportError as e:
        raise ImportError(f"Gerekli kütüphane kurulu değil: {e}")

    dtype = torch.float16 if "cuda" in omnivoice_device else torch.float32
    log.info(f"OmniVoice yükleniyor: {omnivoice_model} @ {omnivoice_device}")
    model = OmniVoice.from_pretrained(
        omnivoice_model,
        device_map=omnivoice_device,
        dtype=dtype,
    )
    log.info("Model hazır.")

    for seg in segments:
        sid = seg["id"]
        final_text = seg.get("final_text", "") or ""
        if not final_text.strip():
            log.warning(f"Seg {sid}: final_text boş, atlanıyor.")
            continue

        ref_audio = seg.get("ref_audio_path")
        out_path  = f"{tmp}/tts_{sid}.wav"
        raw_path  = f"{tmp}/tts_raw_{sid}.wav"

        try:
            audio = model.generate(
                text=seg["final_text"],
                ref_audio=ref_audio,
                ref_text=seg.get("ref_text"),
                language=target_language_id,
                num_step=32,
                speed=1.0,
            )
            sf.write(raw_path, audio[0], 24000)

            trim_audio_silence(raw_path, out_path, threshold_db=silence_threshold)

            if apply_extra_trim and not verify_no_leading_silence(out_path, threshold_db=silence_threshold):
                log.info(f"Seg {sid}: Ekstra sessizlik kırpma uygulanıyor...")
                trim_audio_silence(raw_path, out_path, threshold_db="-35dB")

            tts_dur = get_audio_duration(out_path)
            log.info(f"Seg {sid}: TTS={tts_dur:.2f}s, hedef={seg['duration']:.2f}s")

            seg["tts_output_path"] = out_path
            seg["tts_duration"]    = round(tts_dur, 3)

        except Exception as e:
            log.error(f"Seg {sid}: TTS hatası — {e}")
            seg["warning"] = (seg["warning"] or "") + f" | TTS hatası: {e}"

    save_segments(segments, f"{tmp}/segments.json")
    write_srt(segments, "final_text", f"{tmp}/translated_final.srt")

    log.info("TTS üretimi tamamlandı. Model RAM'de tutuluyor.")
    return segments, model


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 7 — HIZ AYARI
# ══════════════════════════════════════════════════════════════════════════════

def get_atempo_filter(speed_factor: float) -> str:
    if speed_factor == 1.0:
        return ""
    filters = []
    while speed_factor < 0.5:
        filters.append("atempo=0.5")
        speed_factor /= 0.5
    while speed_factor > 100.0:
        filters.append("atempo=100.0")
        speed_factor /= 100.0
    if speed_factor != 1.0:
        filters.append(f"atempo={speed_factor:.4f}")
    return ",".join(filters)


def step7_speed_adjust(
    segments: list,
    omnivoice_model,     # yüklenmiş model nesnesi
    tmp: str,
    max_speed_factor: float = 1.25,
    max_shortening_ratio: float = 0.40,
    silence_threshold: str = "-40dB",
    slow_video: bool = False,
) -> list:
    """
    TTS süresi segment süresini aşan segmentler için:
    slow_video=False (varsayılan):
      1. atempo ≤ max_speed_factor → FFmpeg hızlandırma
      2. atempo > max_speed_factor → OmniVoice duration parametresiyle yeniden üret
    slow_video=True:
      TTS uzun olan segmentlerde sesi hızlandırmak yerine video_slow_factor kaydedilir.
      step8'de video bu oranda yavaşlatılacak.
    """
    log.info("=" * 60)
    log.info(f"ADIM 7 — Hız ayarı (slow_video={'AÇIK' if slow_video else 'KAPALI'})")

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    for seg in segments:
        sid      = seg["id"]
        tts_path = seg.get("tts_output_path")
        tts_dur  = seg.get("tts_duration")
        seg_dur  = seg["duration"]

        if not tts_path or tts_dur is None:
            continue

        # Güvenlik: negatif veya sıfır süre kontrolü
        if seg_dur <= 0.01:
            log.warning(f"Seg {sid}: segment süresi çok kısa veya negatif ({seg_dur:.3f}s), atlanıyor.")
            continue

        # Süre kontrolü
        speed_factor = tts_dur / seg_dur

        if abs(speed_factor - 1.0) < 0.01:
            # Süre neredeyse tam aynı
            final_path = f"{tmp}/tts_{sid}_final.wav"
            shutil.copy(tts_path, final_path)
            seg["tts_output_path"] = final_path
            seg["speed_adjusted"] = False
            continue

        if speed_factor < 1.0:
            # Ses kısa → yavaşlat veya padding ekle (slow_video modu fark etmez)
            log.info(f"Seg {sid}: ses kısa ({tts_dur:.2f}s < {seg_dur:.2f}s), yavaşlatılıyor (speed_factor={speed_factor:.3f})")
            force_omnivoice = speed_factor < max_shortening_ratio
        else:
            # Ses uzun → hızlandır VEYA video yavaşlat
            if slow_video:
                # ── SLOW VIDEO MODU: sesi olduğu gibi bırak, video_slow_factor kaydet ──
                log.info(
                    f"Seg {sid}: ses uzun ({tts_dur:.2f}s > {seg_dur:.2f}s), "
                    f"video yavaşlatılacak (factor={speed_factor:.3f})"
                )
                final_path = f"{tmp}/tts_{sid}_final.wav"
                shutil.copy(tts_path, final_path)
                seg["tts_output_path"] = final_path
                seg["speed_adjusted"] = False
                seg["video_slow_factor"] = round(speed_factor, 4)
                seg["speed_factor"] = round(speed_factor, 4)
                continue
            else:
                log.info(f"Seg {sid}: ses uzun ({tts_dur:.2f}s > {seg_dur:.2f}s), hızlandırılıyor (speed_factor={speed_factor:.3f})")
                force_omnivoice = speed_factor > max_speed_factor

        if not force_omnivoice:
            fast_path  = f"{tmp}/tts_{sid}_fast.wav"
            final_path = f"{tmp}/tts_{sid}_final.wav"
            atempo_str = get_atempo_filter(speed_factor)
            
            run(
                f'ffmpeg -y -i {q(tts_path)} '
                f'-filter:a "{atempo_str}" {q(fast_path)}',
                f"atempo seg {sid}"
            )
            # Kalan boşluğu doldur
            actual = get_audio_duration(fast_path)
            pad    = max(0.0, seg_dur - actual)
            if pad > 0.001:
                run(
                    f'ffmpeg -y -i {q(fast_path)} '
                    f'-af "apad=pad_dur={pad}" {q(final_path)}',
                    f"apad after atempo seg {sid}"
                )
            else:
                shutil.copy(fast_path, final_path)

            seg["tts_output_path"] = final_path
            seg["speed_adjusted"]  = True
            seg["speed_factor"]    = round(speed_factor, 4)

        else:
            # OmniVoice duration ile zorla
            log.warning(
                f"Seg {sid}: speed_factor={speed_factor:.3f} limitleri aştı, "
                f"OmniVoice duration ile yeniden üretiliyor."
            )
            forced_path = f"{tmp}/tts_{sid}_forced.wav"
            forced_ok = False
            try:
                if omnivoice_model is not None and sf is not None:
                    audio = omnivoice_model.generate(
                        text=seg["final_text"],
                        ref_audio=seg.get("ref_audio_path"),
                        ref_text=seg.get("ref_text"),
                        duration=seg_dur,
                        num_step=32,
                    )
                    raw_forced = f"{tmp}/tts_raw_{sid}_forced.wav"
                    sf.write(raw_forced, audio[0], 24000)
                    trim_audio_silence(raw_forced, forced_path, threshold_db=silence_threshold)
                else:
                    # Model yoksa limit faktörü ile zorla
                    limit_factor = max_speed_factor if speed_factor > 1.0 else max_shortening_ratio
                    atempo_str = get_atempo_filter(limit_factor)
                    run(
                        f'ffmpeg -y -i {q(tts_path)} '
                        f'-filter:a "{atempo_str}" {q(forced_path)}',
                        f"forced atempo {limit_factor} seg {sid}"
                    )
                forced_ok = True
            except Exception as e:
                log.error(f"Seg {sid}: forced duration hatası — {e}")
                shutil.copy(tts_path, forced_path)
                seg["warning"] = (seg["warning"] or "") + f" | Forced duration hatası: {e}"

            seg["tts_output_path"] = forced_path
            seg["speed_adjusted"]  = True
            seg["speed_factor"]    = round(speed_factor, 4)
            seg["forced_duration"] = True
            if forced_ok:
                seg["warning"] = (seg["warning"] or "") + \
                    f" | OmniVoice duration ile zorlandı, hız limiti aşıldı."

    save_segments(segments, f"{tmp}/segments.json")
    log.info("Hız ayarı tamamlandı.")
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 8 — VİDEO BİRLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def step8_assemble_video(
    segments: list,
    video_path: str,
    bg_path: str,
    tmp: str,
    output_path: str,
    vocal_volume: float = 0.9,
    bg_volume: float = 1.0,
) -> str:
    """
    1. Tüm TTS segmentlerini zaman çizelgesine yerleştir → dubbed_vocals.wav
    2. Arka plan sesiyle miksleme → final_audio.wav
    3. Orijinal video + final_audio → output_dubbed.mp4
    """
    log.info("=" * 60)
    log.info("ADIM 8 — Video birleştirme")

    video_dur = get_video_duration(video_path)

    # ── 1. Boş taban oluştur
    silent_base = f"{tmp}/silent_base.wav"
    run(
        f'ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono '
        f'-t {video_dur:.3f} {q(silent_base)}',
        "silent base"
    )

    # ── 2. Segment seslerini yerleştir
    valid_segs = [
        s for s in segments
        if s.get("tts_output_path") and Path(s["tts_output_path"]).exists()
    ]

    if not valid_segs:
        log.warning("Hiç geçerli TTS segmenti yok, arka plan sesi kullanılacak.")
        dubbed_vocals = silent_base
    else:
        # ffmpeg filter_complex ile adelay
        inputs   = f"-i {q(silent_base)} "
        filters  = []
        mix_ins  = "[0]"

        for i, seg in enumerate(valid_segs):
            delay_ms = int(seg["start"] * 1000)
            seg_path = seg["tts_output_path"]
            try:
                dur = get_audio_duration(seg_path)
            except Exception:
                dur = seg.get("duration", 1.0)

            # Ardışık segmentler arası boşluk kontrolü
            prev_end = valid_segs[i - 1]["start"] + valid_segs[i - 1].get("duration", 1.0) if i > 0 else -1.0
            next_start = valid_segs[i + 1]["start"] if i < len(valid_segs) - 1 else 9999.0
            gap_before = seg["start"] - prev_end
            gap_after = next_start - (seg["start"] + dur)

            # Sadece gerçek boşluk varsa fade uygula (click önleme: 2ms)
            fade_parts = []
            if gap_before > 0.15 or i == 0:
                fade_parts.append("afade=t=in:ss=0:d=0.002")
            out_st = max(0.0, dur - 0.002)
            if gap_after > 0.15 or i == len(valid_segs) - 1:
                fade_parts.append(f"afade=t=out:st={out_st:.3f}:d=0.002")

            fade_str = ",".join(fade_parts) + "," if fade_parts else ""
            inputs  += f"-i {q(seg_path)} "
            filters.append(f"[{i+1}]{fade_str}adelay={delay_ms}|{delay_ms}[s{i}]")
            mix_ins += f"[s{i}]"

        n_inputs = len(valid_segs) + 1  # base + segmentler
        filter_str = ";".join(filters)
        filter_str += f";{mix_ins}amix=inputs={n_inputs}:duration=first:normalize=0[dubbed]"

        filter_script_path = f"{tmp}/filter_script.txt"
        Path(filter_script_path).write_text(filter_str, encoding="utf-8")

        dubbed_vocals = f"{tmp}/dubbed_vocals.wav"
        run(
            f'ffmpeg -y {inputs} '
            f'-filter_complex_script {q(filter_script_path)} '
            f'-map "[dubbed]" {q(dubbed_vocals)}',
            "dubbed vocals assembly"
        )

    # ── 3. Arka plan + dublajlı vokal miksleme
    final_audio = f"{tmp}/final_audio.wav"
    run(
        f'ffmpeg -y -i {q(bg_path)} -i {q(dubbed_vocals)} '
        f'-filter_complex "[0][1]amix=inputs=2:duration=first:weights={bg_volume} {vocal_volume}[mix]" '
        f'-map "[mix]" {q(final_audio)}',
        "audio mix"
    )

    # ── 4. Video + ses birleştir
    run(
        f'ffmpeg -y -i {q(video_path)} -i {q(final_audio)} '
        f'-c:v copy -c:a aac -b:a 192k '
        f'-map 0:v:0 -map 1:a:0 {q(output_path)}',
        "final video"
    )

    log.info(f"Video tamamlandı: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 8-ALT — VIDEO YAVAŞLATMA MODUNDA BİRLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def step8_assemble_video_slow(
    segments: list,
    video_path: str,
    bg_path: str,
    tmp: str,
    output_path: str,
    vocal_volume: float = 0.9,
    bg_volume: float = 1.0,
) -> str:
    """
    slow_video modu: video_slow_factor > 1.0 olan segmentlerde videoyu yavaşlatır.

    Tek FFmpeg geçişi ile trim + setpts + concat filter kullanır.
    Dosya kesme/birleştirme yapılmaz → kesme artefaktları oluşmaz.
    """
    log.info("=" * 60)
    log.info("ADIM 8 — Video birleştirme (SLOW VIDEO MODU — tek geçiş)")

    video_dur = get_video_duration(video_path)

    # ── 1. Yavaşlatılacak segmentleri belirle ve sırala
    slow_segs = sorted(
        [s for s in segments if s.get("video_slow_factor", 1.0) > 1.01],
        key=lambda s: s["start"]
    )

    if not slow_segs:
        log.info("Yavaşlatılacak segment yok, normal birleştirme kullanılıyor.")
        return step8_assemble_video(
            segments, video_path, bg_path, tmp, output_path,
            vocal_volume, bg_volume,
        )

    log.info(f"{len(slow_segs)} segment için video yavaşlatılacak.")

    # ── 2. Zaman aralıklarını oluştur: (start, end, factor)
    time_ranges = []
    current_time = 0.0

    for sseg in slow_segs:
        seg_start = sseg["start"]
        seg_end = sseg["end"]
        factor = sseg["video_slow_factor"]

        # Normal bölüm (önceki parçadan bu segmente kadar)
        if seg_start > current_time + 0.02:
            time_ranges.append((current_time, seg_start, 1.0))

        # Yavaşlatılmış bölüm
        time_ranges.append((seg_start, seg_end, factor))
        current_time = seg_end

    # Son normal bölüm
    if current_time < video_dur - 0.02:
        time_ranges.append((current_time, video_dur, 1.0))

    log.info(f"  Toplam {len(time_ranges)} zaman aralığı oluşturuldu.")

    # ── 3. Tek geçişli FFmpeg filter_complex oluştur
    video_filters = []
    audio_filters = []
    concat_v_labels = []
    concat_a_labels = []

    for i, (start, end, factor) in enumerate(time_ranges):
        # Video: trim → setpts
        if factor == 1.0:
            video_filters.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]"
            )
        else:
            video_filters.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts={factor:.4f}*(PTS-STARTPTS)[v{i}]"
            )

        # Audio: atrim → asetpts (+ atempo for slow parts)
        if factor == 1.0:
            audio_filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )
        else:
            inv_factor = 1.0 / factor
            atempo_chain = get_atempo_filter(inv_factor)
            audio_filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,{atempo_chain}[a{i}]"
            )

        concat_v_labels.append(f"[v{i}]")
        concat_a_labels.append(f"[a{i}]")

    # Concat filter
    n = len(time_ranges)
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    concat_filter = f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"

    full_filter = ";".join(video_filters + audio_filters) + ";" + concat_filter

    # Filter script dosyasına yaz (çok uzun olabilir)
    filter_script = f"{tmp}/slow_video_filter.txt"
    Path(filter_script).write_text(full_filter, encoding="utf-8")

    # ── 4. Tek geçişte video oluştur
    concat_video = f"{tmp}/concat_video_slow.mp4"
    run(
        f'ffmpeg -y -i {q(video_path)} '
        f'-filter_complex_script {q(filter_script)} '
        f'-map "[outv]" -map "[outa]" '
        f'-c:v libx264 -preset medium -crf 18 -c:a aac -b:a 192k '
        f'{q(concat_video)}',
        "single-pass slow video"
    )

    total_new_dur = get_video_duration(concat_video)
    log.info(f"Yeni video süresi: {total_new_dur:.2f}s (orijinal: {video_dur:.2f}s)")

    # ── 5. Zaman eşleme tablosu oluştur
    time_mapping = []
    new_time = 0.0
    for start, end, factor in time_ranges:
        orig_dur = end - start
        new_dur = orig_dur * factor
        time_mapping.append((start, end, new_time, new_time + new_dur, factor))
        new_time += new_dur

    def map_time(orig_t):
        """Orijinal zamandan yeni zamana dönüştür."""
        for orig_s, orig_e, new_s, new_e, sf in time_mapping:
            if orig_s <= orig_t <= orig_e:
                ratio = (orig_t - orig_s) / max(orig_e - orig_s, 0.001)
                return new_s + ratio * (new_e - new_s)
        return orig_t * (total_new_dur / video_dur)

    # ── 6. TTS seslerini yeni zaman çizelgesine göre yerleştir
    silent_base = f"{tmp}/silent_base_slow.wav"
    run(
        f'ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono '
        f'-t {total_new_dur:.3f} {q(silent_base)}',
        "silent base slow"
    )

    valid_segs = [
        s for s in segments
        if s.get("tts_output_path") and Path(s["tts_output_path"]).exists()
    ]

    if not valid_segs:
        dubbed_vocals = silent_base
    else:
        inputs = f"-i {q(silent_base)} "
        filters = []
        mix_ins = "[0]"

        for i, seg in enumerate(valid_segs):
            new_start = map_time(seg["start"])
            delay_ms = int(new_start * 1000)
            seg_path = seg["tts_output_path"]
            try:
                dur = get_audio_duration(seg_path)
            except Exception:
                dur = seg.get("duration", 1.0)

            # Ardışık segmentler arası boşluk kontrolü
            prev_end = map_time(valid_segs[i - 1]["start"]) + valid_segs[i - 1].get("duration", 1.0) if i > 0 else -1.0
            next_start_mapped = map_time(valid_segs[i + 1]["start"]) if i < len(valid_segs) - 1 else 9999.0
            gap_before = new_start - prev_end
            gap_after = next_start_mapped - (new_start + dur)

            fade_parts = []
            if gap_before > 0.15 or i == 0:
                fade_parts.append("afade=t=in:ss=0:d=0.002")
            out_st = max(0.0, dur - 0.002)
            if gap_after > 0.15 or i == len(valid_segs) - 1:
                fade_parts.append(f"afade=t=out:st={out_st:.3f}:d=0.002")

            fade_str = ",".join(fade_parts) + "," if fade_parts else ""
            inputs += f"-i {q(seg_path)} "
            filters.append(f"[{i+1}]{fade_str}adelay={delay_ms}|{delay_ms}[s{i}]")
            mix_ins += f"[s{i}]"

        n_inputs = len(valid_segs) + 1
        filter_str = ";".join(filters)
        filter_str += f";{mix_ins}amix=inputs={n_inputs}:duration=first:normalize=0[dubbed]"

        filter_script_path = f"{tmp}/filter_script_slow.txt"
        Path(filter_script_path).write_text(filter_str, encoding="utf-8")

        dubbed_vocals = f"{tmp}/dubbed_vocals_slow.wav"
        run(
            f'ffmpeg -y {inputs} '
            f'-filter_complex_script {q(filter_script_path)} '
            f'-map "[dubbed]" {q(dubbed_vocals)}',
            "dubbed vocals assembly slow"
        )

    # ── 7. Arka plan sesini de yeni süreye uzat
    bg_stretched = f"{tmp}/bg_stretched.wav"
    bg_dur = get_audio_duration(bg_path)
    if total_new_dur > bg_dur + 0.1:
        pad_dur = total_new_dur - bg_dur
        run(
            f'ffmpeg -y -i {q(bg_path)} '
            f'-af "apad=pad_dur={pad_dur:.3f}" {q(bg_stretched)}',
            "stretch bg audio"
        )
    else:
        shutil.copy(bg_path, bg_stretched)

    # ── 8. Arka plan + dublajlı vokal miksleme
    final_audio = f"{tmp}/final_audio_slow.wav"
    run(
        f'ffmpeg -y -i {q(bg_stretched)} -i {q(dubbed_vocals)} '
        f'-filter_complex "[0][1]amix=inputs=2:duration=first:weights={bg_volume} {vocal_volume}[mix]" '
        f'-map "[mix]" {q(final_audio)}',
        "audio mix slow"
    )

    # ── 9. Video + ses birleştir
    run(
        f'ffmpeg -y -i {q(concat_video)} -i {q(final_audio)} '
        f'-c:v copy -c:a aac -b:a 192k '
        f'-map 0:v:0 -map 1:a:0 -shortest {q(output_path)}',
        "final video slow"
    )

    log.info(f"Video tamamlandı (slow mode): {output_path}")
    log.info(f"Toplam süre: {total_new_dur:.2f}s (orijinal: {video_dur:.2f}s, fark: +{total_new_dur - video_dur:.2f}s)")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 8.5 — DEBUG SENKRONİZASYON VİDEOSU (İsteğe Bağlı)
# ══════════════════════════════════════════════════════════════════════════════

def step8_debug_sync_video(
    segments: list,
    video_path: str,
    vocals_path: str,
    bg_path: str,
    tmp: str,
    output_path: str,
) -> str:
    """
    Orijinal sesi referans alarak senkronizasyon kontrolü için bir video oluşturur.
    Whisper'ın çıkardığı start/end zamanlarına tam olarak orijinal ses kesilip yerleştirilir.
    """
    log.info("=" * 60)
    log.info("DEBUG ADIMI — Senkronizasyon test videosu oluşturuluyor")

    video_dur = get_video_duration(video_path)
    silent_base = f"{tmp}/silent_base_debug.wav"
    run(
        f'ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono '
        f'-t {video_dur:.3f} {q(silent_base)}',
        "silent base debug"
    )

    inputs   = f"-i {q(silent_base)} "
    filters  = []
    mix_ins  = "[0]"

    valid_count = 0
    for i, seg in enumerate(segments):
        sid = seg["id"]
        start = seg["start"]
        dur = seg["duration"]
        
        # Tam start/end arası kesim yap
        strict_wav = f"{tmp}/strict_orig_{sid}.wav"
        try:
            run(
                f'ffmpeg -y -i {q(vocals_path)} -ss {start:.3f} -t {dur:.3f} '
                f'-acodec pcm_s16le -ar 24000 -ac 1 {q(strict_wav)}',
                f"extract strict {sid}"
            )
            delay_ms = int(start * 1000)
            inputs  += f"-i {q(strict_wav)} "
            filters.append(f"[{valid_count+1}]adelay={delay_ms}|{delay_ms}[s{valid_count}]")
            mix_ins += f"[s{valid_count}]"
            valid_count += 1
        except Exception as e:
            log.warning(f"Debug ses çıkarılamadı (Seg {sid}): {e}")
            continue

    if valid_count == 0:
        log.warning("Hiç geçerli debug segmenti oluşturulamadı.")
        return ""

    n_inputs = valid_count + 1
    filter_str = ";".join(filters)
    filter_str += f";{mix_ins}amix=inputs={n_inputs}:duration=first:normalize=0[dubbed]"

    filter_script_path = f"{tmp}/filter_script_debug.txt"
    Path(filter_script_path).write_text(filter_str, encoding="utf-8")

    dubbed_vocals = f"{tmp}/dubbed_vocals_debug.wav"
    run(
        f'ffmpeg -y {inputs} '
        f'-filter_complex_script {q(filter_script_path)} '
        f'-map "[dubbed]" {q(dubbed_vocals)}',
        "dubbed vocals assembly debug"
    )

    final_audio = f"{tmp}/final_audio_debug.wav"
    run(
        f'ffmpeg -y -i {q(bg_path)} -i {q(dubbed_vocals)} '
        f'-filter_complex "[0][1]amix=inputs=2:duration=first:weights=1 0.9[mix]" '
        f'-map "[mix]" {q(final_audio)}',
        "audio mix debug"
    )

    debug_output = output_path.replace(".mp4", "_debug_sync.mp4")
    run(
        f'ffmpeg -y -i {q(video_path)} -i {q(final_audio)} '
        f'-c:v copy -c:a aac -b:a 192k '
        f'-map 0:v:0 -map 1:a:0 {q(debug_output)}',
        "final debug video"
    )

    log.info(f"Senkronizasyon test videosu tamamlandı: {debug_output}")
    return debug_output


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 9 — ALTYAZI GÖMME (İsteğe Bağlı)
# ══════════════════════════════════════════════════════════════════════════════

def step9_embed_subtitles(
    output_path: str,
    srt_path: str,
    tmp: str,
) -> str:
    """Altyazıyı videoya göm (isteğe bağlı)."""
    log.info("=" * 60)
    log.info("ADIM 9 — Altyazı gömme")

    subs_output = output_path.replace(".mp4", "_with_subs.mp4")
    try:
        run(
            f'ffmpeg -y -i {q(output_path)} -i {q(srt_path)} '
            f'-c copy -c:s mov_text {q(subs_output)}',
            "embed subtitles"
        )
        log.info(f"Altyazılı video: {subs_output}")
        return subs_output
    except Exception as e:
        log.warning(f"Altyazı gömme başarısız: {e}")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 10 — RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def step10_report(segments: list, video_path: str, target_language: str, tmp: str):
    """Pipeline raporu oluştur ve konsola yazdır."""
    log.info("=" * 60)
    log.info("ADIM 10 — Rapor oluşturuluyor")

    total      = len(segments)
    shortened  = sum(1 for s in segments if s.get("needs_shortening"))
    speed_adj  = sum(1 for s in segments if s.get("speed_adjusted"))
    forced_dur = sum(1 for s in segments if s.get("forced_duration"))
    warnings   = [
        f"Segment {s['id']}: {s['warning']}"
        for s in segments if s.get("warning")
    ]

    report = {
        "video_path":              video_path,
        "target_language":         target_language,
        "total_segments":          total,
        "shortened_segments":      shortened,
        "speed_adjusted_segments": speed_adj,
        "forced_duration_segments":forced_dur,
        "warnings":                warnings,
        "segments":                segments,
    }

    report_path = f"{tmp}/pipeline_report.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "═" * 60)
    print("  📊 PIPELINE RAPORU")
    print("═" * 60)
    print(f"  ✅ Toplam segment          : {total}")
    print(f"  ✂️  Kısaltılan              : {shortened}")
    print(f"  ⚡ Hızlandırılan           : {speed_adj}")
    print(f"  🔒 Zorla süresi sabitlenen : {forced_dur}")
    if warnings:
        print(f"  ⚠️  Uyarı sayısı           : {len(warnings)}")
        for w in warnings:
            print(f"       • {w}")
    print("═" * 60 + "\n")

    shutil.copy(report_path, "pipeline_report.json")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video Dublaj Pipeline — Demucs + Whisper + OmniVoice"
    )
    parser.add_argument("--video",                required=True,  help="Giriş video dosyası")
    parser.add_argument("--target_language",      required=True,  help="Hedef dil (ör: Turkish)")
    parser.add_argument("--target_language_id",   required=True,  help="OmniVoice dil kodu (ör: tr)")
    parser.add_argument("--source_language",      default="auto",
                        help="Whisper kaynak dil kodu (ör: en, de, fr). 'auto' = otomatik algıla")
    parser.add_argument("--lm_studio_url",        default="http://localhost:11434/v1",
                        help="Ollama OpenAI-compatible API endpoint (ör: http://localhost:11434/v1)")
    parser.add_argument("--lm_studio_model",      default="gemma4:e2b",
                        help="Kullanılacak model adı (varsayılan: gemma4:e2b)")
    parser.add_argument("--whisper_model",        default="large-v3",
                        help="Whisper modeli (tiny, base, small, medium, large-v3)")
    # GUI geriye uyumluluk — bu argümanlar kabul edilir ama dahili olarak kullanılmaz
    parser.add_argument("--translation_api",      default="ollama",
                        help="(Geriye uyumluluk) Çeviri OpenAI-compatible endpoint üzerinden yapılır (örn. Ollama)")
    parser.add_argument("--translation_api_key",  default="",
                        help="(Geriye uyumluluk) Şu an kullanılmıyor")
    parser.add_argument("--omnivoice_model",      default="k2-fsa/OmniVoice")
    parser.add_argument("--omnivoice_device",     default="cuda:0",
                        help="cuda:0 | mps | cpu")
    parser.add_argument("--output",               default=None,
                        help="Çıktı video dosyası (Varsayılan: [video_adi]_output.mp4)")
    parser.add_argument("--embed_subtitles",      action="store_true",
                        help="Altyazıyı videoya göm")
    parser.add_argument("--debug_sync",           action="store_true",
                        help="Orijinal sesleri kesip zaman tüneline dizerek senkronizasyon test videosu oluşturur")

    parser.add_argument("--keep_tmp",             action="store_true",
                        help="Geçici dosyaları silme")
    parser.add_argument("--no_confirm",           action="store_true",
                        help="TTS adımı öncesi onay sormadan devam et (GUI modu için)")
    # ── Yeni parametreler ──
    parser.add_argument("--max_speed_factor",     type=float, default=1.25,
                        help="Maksimum hızlandırma oranı (varsayılan: 1.25)")
    parser.add_argument("--max_shortening_ratio", type=float, default=0.40,
                        help="Maksimum kısaltma oranı, bunun üzerinde hızlandırma tercih edilir (varsayılan: 0.40)")
    parser.add_argument("--silence_threshold",    default="-40dB",
                        help="OmniVoice ses başı sessizlik kırpma eşiği (varsayılan: -40dB)")
    parser.add_argument("--chars_per_second",     type=float, default=21.0,
                        help="TTS karakter/saniye kalibrasyonu — çeviri uzunluğunu belirler (varsayılan: 21)")
    parser.add_argument("--context_size",          type=int,   default=2,
                        help="Çeviri bağlam penceresi — her segment için kaç önceki/sonraki segment bağlam olarak gönderilir (varsayılan: 2)")

    parser.add_argument("--vocal_volume",         type=float, default=0.9,
                        help="Dublaj ses seviyesi (varsayılan: 0.9)")
    parser.add_argument("--bg_volume",            type=float, default=1.0,
                        help="Arka plan ses seviyesi (varsayılan: 1.0)")
    parser.add_argument("--slow_video",           action="store_true",
                        help="Sesi hızlandırmak yerine videoyu yavaşlat (toplam süre uzar)")
    parser.add_argument("--resume",               action="store_true",
                        help="Önceki çalışmadan kaldığı yerden devam et (tmp dizini varsa)")
    parser.add_argument("--merge_speaker_gap",     type=int, default=0,
                        help="Aynı konuşmacının ardışık segmentlerini birleştirme eşiği (ms). 0=kapalı, ör: 1000=1sn altı boşlukları birleştir")
    args = parser.parse_args()
    
    video_stem = Path(args.video).stem
    if args.output is None:
        args.output = f"{video_stem}_output.mp4"

    # ── Kullanıcı Soruları (Başlangıçta) ──
    apply_extra_trim = True
    translation_strategy = "compress"
    if not args.no_confirm:
        print("\n" + "!" * 60)
        print("  ❓ Süreye sığmayan uzun çeviriler için hangi yöntem kullanılsın?")
        print("     1: Çeviriyi Kısalt (LLM cümleyi özetleyerek kısaltır) [Varsayılan]")
        print("     2: Sesi Hızlandır (Çeviri kısaltılmaz, doğal bırakılır, seslendirme sonradan hızlandırılır)")
        secim = input("  Seçiminiz (1/2): ").strip()
        if secim == '2':
            translation_strategy = "speed_up"
            
        print("\n  ❓ TTS Adımında 'Ekstra Sessizlik Kırpma' (Sıkı Dudak Senkronizasyonu) uygulansın mı?")
        print("     (Evet derseniz, sesin başındaki en ufak boşluklar bile -35dB ile agresif kırpılır.)")
        cevap = input("  Uygulansın mı? (e/h) [Varsayılan: e]: ").lower().strip()
        if cevap == 'h':
            apply_extra_trim = False
        print("!" * 60)

    # ── Geçici dizin
    tmp = f"{video_stem}_tmp"
    resuming = args.resume and Path(tmp).exists() and Path(f"{tmp}/segments.json").exists()

    if resuming:
        log.info("🔄 RESUME MODU — Önceki çalışmadan devam ediliyor.")
        segments = json.loads(Path(f"{tmp}/segments.json").read_text(encoding="utf-8"))
        log.info(f"  {len(segments)} segment yüklendi.")
    else:
        if Path(tmp).exists():
            shutil.rmtree(tmp)
        Path(tmp).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    log.info("═" * 60)
    log.info("  🎬 Video Dublaj Pipeline Başlatıldı")
    log.info(f"  Video       : {args.video}")
    log.info(f"  Hedef dil   : {args.target_language} ({args.target_language_id})")
    log.info(f"  Kaynak dil  : {args.source_language}")
    log.info(f"  Çeviri API  : {args.translation_api}")
    log.info(f"  LLM API     : {args.lm_studio_url} | model={args.lm_studio_model}")
    log.info(f"  OmniVoice   : {args.omnivoice_model} @ {args.omnivoice_device}")
    log.info(f"  Max hız     : {args.max_speed_factor}x | Max kısaltma: {args.max_shortening_ratio:.0%}")
    log.info(f"  Slow Video  : {'AÇIK' if args.slow_video else 'KAPALI'}")
    log.info(f"  Resume      : {'AÇIK' if resuming else 'KAPALI'}")
    log.info("═" * 60)

    # ── Checkpoint kontrol fonksiyonu
    def _check_step_done(step_name, check_fn):
        """Resume modunda adım tamamlanmış mı kontrol et."""
        if not resuming:
            return False
        done = check_fn()
        if done:
            log.info(f"⏭️  {step_name} — zaten tamamlanmış, atlanıyor.")
        return done

    # ADIM 1 — Ses ayırma
    vocals_path = f"{tmp}/vocals.wav"
    bg_path = f"{tmp}/no_vocals.wav"
    if not _check_step_done("ADIM 1 (Ses ayırma)",
                            lambda: Path(vocals_path).exists() and Path(bg_path).exists()):
        vocals_path, bg_path = step1_separate_audio(args.video, tmp)

    # ADIM 2 — Transkripsiyon ve Diarization
    if not _check_step_done("ADIM 2 (Transkripsiyon)",
                            lambda: resuming and segments and segments[0].get("original_text")):
        segments = step2_diarize(
            vocals_path, args.whisper_model, tmp,
            source_language=args.source_language
        )

    # ADIM 2.5 — Konuşmacı Bloklama
    if args.merge_speaker_gap > 0:
        if not _check_step_done("ADIM 2.5 (Konuşmacı Bloklama)",
                                lambda: resuming and segments and segments[0].get("merged_ids")):
            segments = step2_5_merge_speakers(segments, args.merge_speaker_gap, tmp)

    # ADIM 3 — Referans ses segmentleri (sessizlik kırpmalı)
    if not _check_step_done("ADIM 3 (Referans ses)",
                            lambda: resuming and segments and segments[0].get("ref_audio_path")):
        segments = step3_extract_ref_segments(segments, vocals_path, tmp, args.silence_threshold)

    # ADIM 3.5 — Ses Cinsiyeti ve Özel Atama
    if not _check_step_done("ADIM 3.5 (Ses ataması)",
                            lambda: resuming and segments and segments[0].get("voice_source")):
        segments = step3_5_assign_voices(segments, tmp, args.no_confirm)

    # ADIM 4 — Çeviri
    if not _check_step_done("ADIM 4 (Çeviri)",
                            lambda: resuming and segments and segments[0].get("translated_text")):
        segments = step4_translate(
            segments, args.target_language,
            args.lm_studio_url, args.lm_studio_model, tmp,
            translation_strategy=translation_strategy,
            chars_per_second=args.chars_per_second,
            context_size=args.context_size,
        )

    # ADIM 4.5 — TTS Metin Normalizasyonu
    if not _check_step_done("ADIM 4.5 (Normalizasyon)",
                            lambda: resuming and segments and segments[0].get("pre_tts_normalized")):
        segments = step4_5_normalize_for_tts(segments, args.target_language_id, tmp)

    # ADIM 6 — TTS
    model = None
    tts_done = _check_step_done("ADIM 6 (TTS)",
        lambda: resuming and segments and segments[0].get("tts_output_path")
                and Path(segments[0]["tts_output_path"]).exists())

    if not tts_done:
        if not args.no_confirm:
            print("\n" + "!" * 60)
            print("  ⚠️  DİKKAT: OmniVoice TTS (Seslendirme) adımına geçiliyor.")
            print("  GPU RAM'ini boşaltmak için (LLM/Ollama modelini kapatmak vb.) bu aşamada bekleyebilirsiniz.")
            print("!" * 60)

            devam = input("  Devam etmek istiyor musunuz? (e/h): ").lower().strip()
            if devam != 'e':
                log.info("İşlem kullanıcı tarafından durduruldu.")
                return
        else:
            log.info("TTS adımına otomatik geçiliyor (--no_confirm).")

        segments, model = step6_tts(
            segments, args.omnivoice_model, args.omnivoice_device,
            args.target_language_id, tmp,
            silence_threshold=args.silence_threshold,
            apply_extra_trim=apply_extra_trim,
        )

    # ADIM 6.5 — Ton Rengi Dönüşümü (OpenVoice)
    if not _check_step_done("ADIM 6.5 (Ton dönüşümü)",
                            lambda: resuming and not any(s.get("tone_convert") and not s.get("tts_output_path", "").endswith("_tc.wav") for s in segments)):
        segments = step6_5_tone_convert(
            segments, tmp, device=args.omnivoice_device,
        )

    # ADIM 7 — Hız ayarı
    if not _check_step_done("ADIM 7 (Hız ayarı)",
        lambda: resuming and segments and any(
            s.get("speed_adjusted") is not None or s.get("video_slow_factor")
            for s in segments if s.get("tts_output_path")
        )):
        segments = step7_speed_adjust(
            segments, omnivoice_model=model, tmp=tmp,
            max_speed_factor=args.max_speed_factor,
            max_shortening_ratio=args.max_shortening_ratio,
            silence_threshold=args.silence_threshold,
            slow_video=args.slow_video,
        )

    # Modelleri bellekten boşalt (Adım 7'den sonra)
    if model is not None:
        del model
        clear_gpu_cache()

    # ADIM 8 — Video birleştirme
    if args.slow_video:
        output = step8_assemble_video_slow(
            segments, args.video, bg_path, tmp, args.output,
            vocal_volume=args.vocal_volume,
            bg_volume=args.bg_volume,
        )
    else:
        output = step8_assemble_video(
            segments, args.video, bg_path, tmp, args.output,
            vocal_volume=args.vocal_volume,
            bg_volume=args.bg_volume,
        )

    # ADIM 8.5 — Debug Senkronizasyon Videosu
    if args.debug_sync:
        step8_debug_sync_video(
            segments, args.video, vocals_path, bg_path, tmp, args.output
        )

    # ADIM 9 — Altyazı gömme (isteğe bağlı)
    srt_final = f"{tmp}/translated_final.srt"
    if args.embed_subtitles and Path(srt_final).exists():
        output = step9_embed_subtitles(output, srt_final, tmp)

    # Çıktı dosyalarını kopyala
    for fname in ["original.srt", "translated_final.srt"]:
        src = Path(f"{tmp}/{fname}")
        if src.exists():
            shutil.copy(src, fname)

    # ADIM 10 — Rapor
    step10_report(segments, args.video, args.target_language, tmp)

    # Geçici dosyaları temizle
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("Geçici dosyalar temizlendi.")

    elapsed = time.time() - start_time
    print(f"  📁 Çıktı video : {output}")
    print(f"  ⏱️  Toplam süre : {elapsed:.1f} saniye\n")


if __name__ == "__main__":
    main()