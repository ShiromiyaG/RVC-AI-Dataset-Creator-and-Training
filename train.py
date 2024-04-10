import contextlib
from multiprocessing import Process
from yt_dlp import YoutubeDL
import gettext
import click
import os
import sys
import subprocess
from glob import glob
from rvccli import run_preprocess_script, run_extract_script, run_train_script
from musicsouceseparationtraining import proc_file
from shiromiyautils import process_spectrogram
import audiofile as af
from uvr import models
import torch
from uvr import models
from pydub import AudioSegment
import json
import shutil
import zipfile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

gettext.bindtextdomain('RVCAIMaker', 'locale')
gettext.textdomain('RVCAIMaker')
_ = gettext.gettext

@contextlib.contextmanager
def suppress_output(supress=True):
    if supress:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        pass

def get_last_modified_file(directory, filter=''):
  arquivos = glob(directory + "/*")
  if filter != '':
      arquivos = [arquivo for arquivo in arquivos if filter in arquivo]
  if arquivos:
      return max(arquivos, key=os.path.getmtime)
  else:
      return None

def backup(model_name, autosave_folder, autosave_minutes_timer, delete_old_weight_and_G_D_files, p1):
    # Definir a pasta de origem e destino
    origem = f'/content/RVC_CLI/logs/{model_name}'
    destino = autosave_folder

    # Definir os prefixos dos arquivos
    prefixo_variavel = 'model_name'  # Substitua com o prefixo real
    prefixo_g = 'G'
    prefixo_d = 'D'

    class FileEventHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            arquivo = os.path.basename(event.src_path)
            if arquivo.startswith(prefixo_variavel) or arquivo.startswith(prefixo_g) or arquivo.startswith(prefixo_d):
                origem_completa = event.src_path
                destino_completo = os.path.join(destino, arquivo)
                shutil.copy(origem_completa, destino_completo)
                print(f'File {arquivo} copied to Google Drive.')

        def on_modified(self, event):
            if event.is_directory:
                return
            arquivo = os.path.basename(event.src_path)
            if arquivo.startswith(prefixo_variavel) or arquivo.startswith(prefixo_g) or arquivo.startswith(prefixo_d):
                origem_completa = event.src_path
                destino_completo = os.path.join(destino, arquivo)
                shutil.copy(origem_completa, destino_completo)
                print(f'File {arquivo} updated in Google Drive.')

    def delete_old_files():
        if delete_old_weight_and_G_D_files:
            for arquivo in os.listdir(destino):
                if arquivo.startswith(prefixo_g) or arquivo.startswith(prefixo_d):
                    arquivo_completo = os.path.join(destino, arquivo)
                    os.remove(arquivo_completo)
                    print(f'File {arquivo} deleted from Google Drive.')

    event_handler = FileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, origem, recursive=False)
    observer.start()

    try:
        while p1.is_alive():
            delete_old_files()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def zip(zip_file_name, folder_path):
  with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
    # Iterate over all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)

            # Add the file to the zip archive
            zip_file.write(file_path, os.path.relpath(file_path, folder_path))

def train(nome_do_modelo, rvc_version, overtraining_detector, overtraining_threshold, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sampling_rate, batch_size, gpu, pitch_guidance, pretrained, custom_pretrained, g_pretrained, d_pretrained):
  run_train_script(nome_do_modelo, rvc_version, overtraining_detector, overtraining_threshold, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sampling_rate, batch_size, gpu, pitch_guidance, pretrained, custom_pretrained, g_pretrained, d_pretrained)
  return

@click.group()
def cli():
    pass

@click.command("download_yt")
@click.option('--link')
def download_yt(link):
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '256',
        }],
        'outtmpl': '/content/musicas/arquivos-originais/%(title)s.%(ext)s',
        'quiet': True
    }
    print(_("Downloading Youtube music..."))
    with suppress_output():
        with YoutubeDL(options) as ydl:
            ydl.download([link])
    print(_("Download of Youtube music complete!"))

@click.command("download_deezer")
@click.option('--link')
@click.option('--bf_secret')
@click.option('--track_url_key')
@click.option('--arl')
@click.option('--supress')
def download_deezer(link, bf_secret, track_url_key, arl, supress):
    print(_("Downloading Deezer music..."))
    with suppress_output(supress):
        with open('/content/OrpheusDL/config/settings.json', 'r') as file:
            data = json.load(file)
        data['modules']['deezer']['bf_secret'] = bf_secret
        data['modules']['deezer']['track_url_key'] = track_url_key
        data['modules']['deezer']['arl'] = arl
        with open('/content/OrpheusDL/config/settings.json', 'w') as file:
            json.dump(data, file, indent=4)
        subprocess.run(["python", "OrpheusDL/orpheus.py", link])
    print(_("Download of Deezer complete!"))

@click.command("separate_vocals")
@click.option('--input_file')
@click.option('--vocal_ensemble')
@click.option('--algorithm_ensemble_vocals')
@click.option('--no_inst_folder')
@click.option('--no_back_folder')
@click.option('--no_noise')
@click.option('--audio_output')
@click.option('--output_folder')
@click.option('--device')
@click.option('--supress')
def separate_vocals(input_file, Vocals_Ensemble, algorithm_ensemble_vocals, no_inst_folder, no_back_folder, output_folder, no_noise, audio_output, device, supress):
    print(_("Separating vocals..."))
    with suppress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # Conevert mp3 to flac
        if input_file.endswith(".mp3"):
            flac_filename = os.path.splitext(input_file)[0] + '.flac'
            if not os.path.exists(flac_filename):
                audio = AudioSegment.from_mp3(input_file)
                audio.export(f"{flac_filename}", format="flac")
                os.remove(input_file)
                input_file = flac_filename
        # MDX23C-8KFFT-InstVoc_HQ
        MDX23C_args = [
            "--model_type", "mdx23c",
            "--config_path", "Music-Source-Separation-Training/models/model_2_stem_full_band_8k.yaml",
            "--start_check_point", "Music-Source-Separation-Training/models/MDX23C-8KFFT-InstVoc_HQ.ckpt",
            "--input_file", f"{input_file}",
            "--store_dir", f"{no_inst_folder}",
        ]
        proc_file(MDX23C_args)
    print(_(f"{basename} processing with MDX23C-8KFFT-InstVoc_HQ is over!"))
    with suppress_output(supress):
        # Ensemble Vocals
        if Vocals_Ensemble:
            lista = []
            lista.append(get_last_modified_file(no_inst_folder, "Vocals"))
            BSRoformer_args = [
                "--model_type", "bs_roformer",
                "--config_path", "Music-Source-Separation-Training/models/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                "--start_check_point", "Music-Source-Separation-Training/models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--input_file", f"{input_file}",
                "--store_dir", f"{no_inst_folder}",
            ]
            proc_file(BSRoformer_args)
            print(_(f"{basename} processing with BSRoformer is over!"))
            lista.append(get_last_modified_file(no_inst_folder, "Vocals"))
            ensemble_voc = os.path.join(no_inst_folder, f"{basename}_ensemble1.wav")
            First_Ensemble_args = [
                "--audio_input", f"{lista[0]}", f"{lista[1]}",
                "--algorithm", f"{algorithm_ensemble_vocals}",
                "--is_normalization", "False",
                "--wav_type_set", "PCM_16"
                "--save_path", f"{ensemble_voc}"
            ]
            process_spectrogram(First_Ensemble_args)
        filename_path = get_last_modified_file(no_inst_folder)
        no_inst_output = os.path.join(no_inst_folder, filename_path)
    with suppress_output(supress):
        # karokee_4band_v2_sn
        Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
        with suppress_output():
            res = Vr(no_inst_output)
            vocals = res["vocals"]
            af.write(f"{no_back_folder}/{basename}_karokee_4band_v2_sn.wav", vocals, Vr.sample_rate)
        torch.cuda.empty_cache()
        filename_path = get_last_modified_file(no_back_folder)
        no_back_output = os.path.join(no_back_folder, filename_path)
    print(_(f"{basename} processing with karokee_4band_v2_sn is over!"))
    with suppress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 384,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        with suppress_output():
            res = MDX(no_back_output)
            no_reverb = res["no reverb"]
            af.write(f"{output_folder}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
        torch.cuda.empty_cache()
    print(_(f"{basename} processing with Reverb HQ is over!"))
    print(_("Vocal processing completed."))
    print(_("Separation complete!"))
    with suppress_output(supress):
        input_for_effects = get_last_modified_file(output_folder)
        if os.path.exists("/content/noise_profile_file"):
            os.remove("/content/noise_profile_file")
        basename = os.path.basename(input_for_effects)
        subprocess.run(["sox", f"{input_for_effects}", "-n", "noiseprof", "noise_profile_file"])
        output_file_noise = f"{no_noise}/{basename}" + "_noise_reduced" + os.path.splitext(input_for_effects)[1]
        subprocess.run(["sox", f"{input_for_effects}", f"{output_file_noise}", "noisered", "noise_profile_file", "0.30"])
        output_file_noise = get_last_modified_file(no_noise)
        subprocess.run(["ffmpeg-normalize", f"{output_file_noise}", "-ar", "44100", "-ext", "wav", "-o", f"{audio_output}/{basename}_normalized.wav", "-t", "-15", "-lrt", "10"])

@click.command("preprocess_train")
@click.option('--model_name')
@click.option('--dataset_path')
@click.option('--sampling_rate')
@click.option('--supress')
def preprocess_train(model_name, dataset_path, sampling_rate, supress):
    print(_("Preprocessing..."))
    with suppress_output(supress):
        run_preprocess_script(model_name, dataset_path, sampling_rate)
    print(_("Preprocessing complete!"))

@click.command("extract_train")
@click.option('--model_name')
@click.option('--rvc_version')
@click.option('--f0method')
@click.option('--hop_length')
@click.option('--sampling_rate')
@click.option('--zip_file_name')
@click.option('--folder_path')
@click.option('--supress')
def extract_train(model_name, rvc_version, f0method, hop_length, sampling_rate, zip_file_name, folder_path, supress):
    print(_("Extracting..."))
    with suppress_output(supress):
        run_extract_script(model_name, rvc_version, f0method, hop_length, sampling_rate)
        zip(zip_file_name, folder_path)
    print(_("Extraction complete!"))

@click.command("training")
@click.option('--autosave')
@click.option('--model_name')
@click.option('--rvc_version')
@click.option('--overtraining_detector')
@click.option('--overtraining_threshold')
@click.option('--save_every_epoch')
@click.option('--save_only_latest')
@click.option('--save_every_weights')
@click.option('--total_epoch')
@click.option('--sampling_rate')
@click.option('--batch_size')
@click.option('--gpu')
@click.option('--pitch_guidance')
@click.option('--pretrained')
@click.option('--custom_pretrained')
@click.option('--g_pretrained')
@click.option('--d_pretrained')
@click.option('--autosave_folder')
@click.option('--delete_old_weight_and_G_D_files')
@click.option('--supress')
def training(autosave, model_name, rvc_version, overtraining_detector, overtraining_threshold, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sampling_rate, batch_size, gpu, pitch_guidance, pretrained, custom_pretrained, g_pretrained, d_pretrained, autosave_folder, delete_old_weight_and_G_D_files, supress):
    with suppress_output(supress):
        if autosave:
            p1 = Process(target = train, args=(model_name, rvc_version, overtraining_detector, overtraining_threshold, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sampling_rate, batch_size, gpu, pitch_guidance, pretrained, custom_pretrained, g_pretrained, d_pretrained))
            p1.start()
            p2 = Process(target = backup, args=(model_name, autosave_folder, delete_old_weight_and_G_D_files, p1))
            p2.start()
            p1.join()
            p2.join()
        else:
            train()

@click.command("choose_pretrain")
@click.option('--model_name')
@click.option('--sample_rate')
@click.option('--supress')
def choose_pretrain(model_name, sample_rate, supress):
    with suppress_output(supress):
        if model_name == "RVC v2":
            if sample_rate == 32000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/D32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D32k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/G32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G32k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0D32k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0G32k.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0G32k.pth"
                    "/content/pretrained_v2/f0D32k.pth" ]
                return pretrained
            if sample_rate == 40000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/D40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D40k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/G40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G40k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0D40k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0G04k.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0G40k.pth"
                    "/content/pretrained_v2/f0D40k.pth" ]
                return pretrained
            if sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/D48k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D48k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/G48k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G48k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D348k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0D48k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G48k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0G48k.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0G48k.pth"
                    "/content/pretrained_v2/f0D48k.pth" ]
                return pretrained
        if model_name == "OV2":
            if sample_rate == 32000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/Ov2Super/f0Ov2Super32kD.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0Ov2Super32kD.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/Ov2Super/f0Ov2Super32kG.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0Ov2Super32kG.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0Ov2Super32kG.pth"
                    "/content/pretrained_v2/f0Ov2Super32kD.pth" ]
                return pretrained
            if sample_rate == 40000 or sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/Ov2Super/f0Ov2Super40kD.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0Ov2Super40kD.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/Ov2Super/f0Ov2Super40kG.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0Ov2Super40kG.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0Ov2Super40kG.pth"
                    "/content/pretrained_v2/f0Ov2Super40kD.pth" ]
                return pretrained
        if model_name == "Rin E3":
            if sample_rate == 32000 or sample_rate == 40000 or sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/RIN/f0RIN40kD.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0RIN40kD.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_custom/RIN/f0RIN40kG.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "f0RIN40kG.pth"])
                pretrained = [
                    "/content/pretrained_v2/f0RIN40kG.pth"
                    "/content/pretrained_v2/f0RIN40kD.pth" ]
                return pretrained
        if model_name == "Itaila":
            if sample_rate == 32000 or sample_rate == 40000 or sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/TheStinger/itaila/resolve/main/ItaIla_32k_D.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "ItaIla_32k_D.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/TheStinger/itaila/resolve/main/ItaIla_32k_G.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "ItaIla_32k_G.pth"])
                pretrained = [
                    "/content/pretrained_v2/ItaIla_32k_G.pth"
                    "/content/pretrained_v2/ItaIla_32k_D.pth" ]
                return pretrained
        if model_name == "SnowieV3":
            if sample_rate == 32000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-32k/resolve/main/D_SnowieV3.1_32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D_SnowieV3.1_32k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-32k/resolve/main/G_SnowieV3.1_32k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G_SnowieV3.1_32k.pth"])
                pretrained = [
                    "/content/pretrained_v2/G_SnowieV3.1_32k.pth"
                    "/content/pretrained_v2/D_SnowieV3.1_32k.pth" ]
                return pretrained
            if sample_rate == 40000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/main/D_SnowieV3.1_40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D_SnowieV3.1_40k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-40k/resolve/main/G_SnowieV3.1_40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G_SnowieV3.1_40k.pth"])
                pretrained = [
                    "/content/pretrained_v2/G_SnowieV3.1_40k.pth"
                    "/content/pretrained_v2/D_SnowieV3.1_40k.pth" ]
                return pretrained
            if sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-48k/resolve/main/D_SnowieV3.1_48k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D_SnowieV3.1_48k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-48k/resolve/main/G_SnowieV3.1_48k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G_SnowieV3.1_48k.pth"])
                pretrained = [
                    "/content/pretrained_v2/G_SnowieV3.1_48k.pth"
                    "/content/pretrained_v2/D_SnowieV3.1_48k.pth" ]
                return pretrained
        if model_name == "SnowieV3 X RIN_E3":
            if sample_rate == 32000 or sample_rate == 40000 or sample_rate == 48000:
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-X-RinE3-40K/resolve/main/D_Snowie-X-Rin_40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "D_Snowie-X-Rin_40k.pth"])
                subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/MUSTAR/SnowieV3.1-X-RinE3-40K/resolve/main/G_Snowie-X-Rin_40k.pth", "-d", "/content/RVC_CLI/pretrained_v2", "-o", "G_Snowie-X-Rin_40k.pth"])
                pretrained = [
                    "/content/pretrained_v2/G_Snowie-X-Rin_40k.pth"
                    "/content/pretrained_v2/D_Snowie-X-Rin_40k.pth" ]
                return pretrained

@click.command("japonese_hubert")
@click.option('--supress')
def japonese_hubert(subress):
    with suppress_output(supress):
        subprocess.run(["wget","-O", "/content/RVC_CLI/hubert_base.pt", "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt"])

def main():
    cli.add_command(download_yt)
    cli.add_command(download_deezer)
    cli.add_command(preprocess_train)
    cli.add_command(separate_vocals)
    cli.add_command(extract_train)
    cli.add_command(training)
    cli.add_command(choose_pretrain)
    cli.add_command(japonese_hubert)
    cli.add_command(train)
    cli()

if __name__ == "__main__":
    main()
