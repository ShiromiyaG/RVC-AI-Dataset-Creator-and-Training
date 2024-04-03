from pydub import AudioSegment, silence
import zipfile
import os
import argparse

def remove_silence(audio):
    # Remove silence
    non_silent_audio = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-40)
    return non_silent_audio

def join_audio_segments(segments, segment_duration):
    joined_segments = []
    current_segment = None
    for segment in segments:
        if current_segment is None:
            current_segment = segment
        elif current_segment.duration_seconds < segment_duration:
            current_segment += segment
        else:
            joined_segments.append(current_segment)
            current_segment = segment

    if current_segment is not None:
        if current_segment.duration_seconds < segment_duration:
            if joined_segments:
                joined_segments[-1] += current_segment
            else:
                joined_segments.append(current_segment)
        else:
            joined_segments.append(current_segment)

    return joined_segments

def split_audio_files(input_folder, segment_duration, zip_file_name):
    audio_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.wav') or f.endswith('.mp3')]

    if not audio_files:
        print("No audio files found in the specified folder.")
        return

    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for i, audio_file_path in enumerate(audio_files):
            audio = AudioSegment.from_file(audio_file_path)

            # Check if the file is in mp3 format
            is_mp3 = False
            if audio_file_path.lower().endswith('.mp3'):
                is_mp3 = True

            # Remove silence
            non_silent_audio = remove_silence(audio)

            segments = []
            for j, segment in enumerate(non_silent_audio):
                if segment.duration_seconds >= segment_duration:
                    segments.extend(segment[0:segment_duration * 1000] for segment in segment[::segment_duration * 1000])
                else:
                    segments.append(segment)

            # Join segments less than the specified duration with the nearest audio segment
            joined_segments = join_audio_segments(segments, segment_duration)

            for k, segment in enumerate(joined_segments):
                segment_file_name = f"segment_{i + 1}_{k + 1}.wav" if not is_mp3 else f"segment_{i + 1}_{k + 1}.mp3"
                segment.export(segment_file_name, format="wav" if not is_mp3 else "mp3")
                zip_file.write(segment_file_name)
                os.remove(segment_file_name)

    print(f"Audio segments have been saved to {zip_file_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio files into segments.")
    parser.add_argument("input_folder", help="Path to the folder containing audio files")
    parser.add_argument("segment_duration", type=int, default=10, help="Duration of each segment in seconds")
    parser.add_argument("--zip_file_name", default="audio_segments.zip", help="Name of the output zip file (default: audio_segments.zip)")

    args = parser.parse_args()

    split_audio_files(args.input_folder, args.segment_duration, args.zip_file_name)
