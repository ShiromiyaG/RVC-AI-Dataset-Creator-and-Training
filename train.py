import contextlib
from multiprocessing import Process
from yt_dlp import YoutubeDL
import gettext
import click
import os
import sys
import subprocess
from glob import glob
from Music_Source_Separation_Training.inference import proc_file
from RVC_CLI.main import run_preprocess_script, run_extract_script, run_train_script
from Utils.spectograma import process_spectrogram
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

@contextlib.contextmanager
def suppress_output():
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
            
            
@click.group()
def cli():
    pass

@click.command("download_yt")
@click.argument('--link')
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
    with suppress_output():
        print(_("Downloading Youtube music..."))
        with YoutubeDL(options) as ydl:
            ydl.download([link])
        print(_("Download of Youtube music complete!"))
 
@click.command("download_deezer")
@click.argument('--link')
@click.argument('--bf_secret')
@click.argument('--track_url_key')
@click.argument('--arl')
def download_deezer(link, bf_secret, track_url_key, arl):
    with suppress_output():
        with open('/content/OrpheusDL/config/settings.json', 'r') as file:
            data = json.load(file)
        data['modules']['deezer']['bf_secret'] = bf_secret
        data['modules']['deezer']['track_url_key'] = track_url_key
        data['modules']['deezer']['arl'] = arl
        with open('/content/OrpheusDL/config/settings.json', 'w') as file:
            json.dump(data, file, indent=4)
        print(_("Downloading Deezer music..."))
        subprocess.run(["python", "OrpheusDL/orpheus.py", link])
        print(_("Download of Deezer complete!"))
        
@click.command("separate_vocals")
@click.argument('--input_file') 
@click.argument('--vocal_ensemble')
@click.argument('--algorithm_ensemble_vocals')
@click.argument('--no_inst_folder') 
@click.argument('--no_back_folder')
@click.argument('--no_noise')
@click.argument('--audio_output')
@click.argument('--output_folder') 
@click.argument('--device')
def separate_vocals(input_file, Vocals_Ensemble, algorithm_ensemble_vocals, no_inst_folder, no_back_folder, output_folder, no_noise, audio_output, device):
    print(_("Separating vocals..."))
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
    with suppress_output():
        proc_file(MDX23C_args)
    print(_(f"{basename} processing with MDX23C-8KFFT-InstVoc_HQ is over!"))
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
        with suppress_output():
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
@click.argument('--model_name')
@click.argument('--dataset_path')
@click.argument('--sampling_rate')
def preprocess_train(model_name, dataset_path, sampling_rate):
    print(_("Preprocessing..."))
    run_preprocess_script(model_name, dataset_path, sampling_rate)
    print(_("Preprocessing complete!"))
    
@click.command("extract_train")
@click.argument('--model_name')
@click.argument('--rvc_version')
@click.argument('--f0method')
@click.argument('--hop_length')
@click.argument('--sampling_rate')
@click.argument('--zip_file_name')
@click.argument('--folder_path')
def extract_train(model_name, rvc_version, f0method, hop_length, sampling_rate, zip_file_name, folder_path):
    print(_("Extracting..."))
    run_extract_script(model_name, rvc_version, f0method, hop_length, sampling_rate)
    zip(zip_file_name, folder_path)
    print(_("Extraction complete!"))
    
@click.command("training")
@click.argument('--autosave')
@click.argument('--model_name')
@click.argument('--rvc_version')
@click.argument('--overtraining_detector')
@click.argument('--overtraining_threshold')
@click.argument('--save_every_epoch')
@click.argument('--save_only_latest')
@click.argument('--save_every_weights')
@click.argument('--total_epoch')
@click.argument('--sampling_rate')
@click.argument('--batch_size')
@click.argument('--gpu')
@click.argument('--pitch_guidance')
@click.argument('--pretrained')
@click.argument('--custom_pretrained')
@click.argument('--g_pretrained')
@click.argument('--d_pretrained')
@click.argument('--autosave_folder')
@click.argument('--delete_old_weight_and_G_D_files')
def training(autosave, model_name, rvc_version, overtraining_detector, overtraining_threshold, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sampling_rate, batch_size, gpu, pitch_guidance, pretrained, custom_pretrained, g_pretrained, d_pretrained, autosave_folder, delete_old_weight_and_G_D_files):
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
@click.argument('--model_name')
@click.argument('--sample_rate')
def choose_pretrain(model_name, sample_rate):
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
def japonese_hubert(): 
    subprocess.run(["wget","-O", "hubert_base.pt", "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt"])
    