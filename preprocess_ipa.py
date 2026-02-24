"""
텍스트를 IPA로 변환하는 전처리 스크립트

입력 filelist: audio_path|text (일반 텍스트)
출력 filelist: audio_path|ipa_text (IPA 변환된 텍스트)
"""
import argparse
from phonemizer import phonemize
from tqdm import tqdm


def preprocess_filelist(input_file, output_file, language='en-us', verbose=True):
    """
    Filelist의 텍스트를 IPA로 변환

    Args:
        input_file: 입력 filelist 경로 (audio_path|text)
        output_file: 출력 filelist 경로 (audio_path|ipa_text)
        language: 언어 코드
            - 'ko': 한국어
            - 'en-us': 영어 (미국)
            - 'cmn': 중국어
            - 'ja': 일본어
            - 'es': 스페인어
            - 'fr': 프랑스어
            - 'de': 독일어
        verbose: 진행상황 출력 여부
    """
    print(f"📂 Reading: {input_file}")
    print(f"🌐 Language: {language}")

    # 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 Total lines: {len(lines)}")

    # 변환
    converted_lines = []
    failed_lines = []

    iterator = tqdm(lines, desc="Converting to IPA") if verbose else lines

    for i, line in enumerate(iterator):
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')
        if len(parts) != 2:
            print(f"⚠️  Line {i+1}: Invalid format (expected 2 columns): {line}")
            failed_lines.append((i+1, line, "Invalid format"))
            continue

        audio_path, text = parts

        try:
            # 텍스트 → IPA 변환
            ipa_text = phonemize(text, language=language, backend='espeak', strip=True)

            if not ipa_text:
                print(f"⚠️  Line {i+1}: Empty IPA conversion for: {text}")
                failed_lines.append((i+1, line, "Empty IPA"))
                continue

            converted_lines.append(f"{audio_path}|{ipa_text}")

        except Exception as e:
            print(f"❌ Line {i+1}: Conversion failed for '{text}': {e}")
            failed_lines.append((i+1, line, str(e)))
            continue

    # 결과 저장
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')

    # 요약
    print(f"\n{'='*70}")
    print(f"✅ Conversion Summary")
    print(f"{'='*70}")
    print(f"Total lines:     {len(lines)}")
    print(f"Converted:       {len(converted_lines)} ({len(converted_lines)/len(lines)*100:.1f}%)")
    print(f"Failed:          {len(failed_lines)} ({len(failed_lines)/len(lines)*100:.1f}%)")

    if failed_lines:
        print(f"\n⚠️  Failed lines:")
        for line_num, original, reason in failed_lines[:10]:  # 처음 10개만 출력
            print(f"  Line {line_num}: {reason}")
            print(f"    {original}")
        if len(failed_lines) > 10:
            print(f"  ... and {len(failed_lines) - 10} more")

    print(f"\n✅ Done! Output saved to: {output_file}")

    return len(converted_lines), len(failed_lines)


def preview_conversion(input_file, language='en-us', num_samples=5):
    """
    변환 결과 미리보기

    Args:
        input_file: 입력 filelist 경로
        language: 언어 코드
        num_samples: 미리볼 샘플 개수
    """
    print(f"{'='*70}")
    print(f"Preview Mode - First {num_samples} samples")
    print(f"{'='*70}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:num_samples]

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')
        if len(parts) != 2:
            print(f"\n❌ Sample {i}: Invalid format")
            continue

        audio_path, text = parts

        try:
            ipa_text = phonemize(text, language=language, backend='espeak', strip=True)

            print(f"\n📝 Sample {i}:")
            print(f"  Original: {text}")
            print(f"  IPA:      {ipa_text}")

        except Exception as e:
            print(f"\n❌ Sample {i}: Conversion failed")
            print(f"  Original: {text}")
            print(f"  Error:    {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text to IPA in filelist')
    parser.add_argument('-i', '--input', required=True, help='Input filelist path (audio_path|text)')
    parser.add_argument('-o', '--output', required=True, help='Output filelist path (audio_path|ipa_text)')
    parser.add_argument('-l', '--language', default='en-us',
                       help='Language code (ko, en-us, cmn, ja, es, fr, de, etc.)')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Preview mode: show first 5 samples without saving')
    parser.add_argument('-n', '--num-samples', type=int, default=5,
                       help='Number of samples to preview (default: 5)')

    args = parser.parse_args()

    if args.preview:
        # 미리보기 모드
        preview_conversion(args.input, args.language, args.num_samples)
    else:
        # 전체 변환
        preprocess_filelist(args.input, args.output, args.language)
