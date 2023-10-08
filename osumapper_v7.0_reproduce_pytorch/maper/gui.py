import gradio as gr
import argparse
from setup_colab import load_pretrained_model
from act_gan_torch_clean import step6_set_gan_params, step6_run_all
from act_newmap_prep import step4_read_new_map_gui
import numpy as np
from timing import get_timing
import re

# pytorch
# from act_rhythm_calc_torch_clean import step5_load_model, step5_load_npz, step5_predict_notes, step5_convert_sliders, step5_save_predictions, step5_set_params

# tensorflow
from act_rhythm_calc import step5_load_model, step5_load_npz, step5_predict_notes, step5_convert_sliders, step5_save_predictions, step5_set_params

from act_modding import step7_modding
from act_final import step8_save_osu_file_gui
# from act_taiko_hitsounds import step8_taiko_hitsounds_set_params, step8_apply_taiko_hitsounds

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Launch in colab")
cmd_opts = parser.parse_args()

def step1(models, map_file, audio_file, dist_multiplier, note_density, slider_favor, divisor_favor, slider_max_ticks):
    global model_params
    # model_params = load_pretrained_model("torchtest")
    model_params = load_pretrained_model(models)

    # step4_read_new_map(uploaded_osu_name)
    step4_read_new_map_gui(map_file.name, audio_file.name)
    
    
    model = step5_load_model(model_file=model_params["rhythm_model"])
    npz = step5_load_npz()
    # params = model_params["rhythm_param"]
    # Or set the parameters here...
    params = step5_set_params(dist_multiplier=dist_multiplier, note_density=note_density, slider_favor=slider_favor, divisor_favor=[divisor_favor] * 4, slider_max_ticks=slider_max_ticks)
    
    predictions = step5_predict_notes(model, npz, params)
    converted = step5_convert_sliders(predictions, params)
    
    step5_save_predictions(converted)
    return "{} notes predicted.".format(np.sum(predictions[0]))

def step2(good_epoch, max_epoch, note_distance_basis, max_ticks_for_ds, next_from_slider_end, box_loss_border, box_loss_value, box_loss_weight, g_epochs, g_batch, g_input_size, c_epochs, c_true_batch, c_false_batch, c_randfalse_batch):
    global osu_a, data

    if next_from_slider_end == "False":
        next_from_slider_end = ""

    # GAN_PARAMS = model_params["gan"]
    # Or manually set the parameters...
    GAN_PARAMS = {
        "divisor" : 4,
        "good_epoch" : int(good_epoch),
        "max_epoch" : int(max_epoch),
        "note_group_size" : 10,
        "g_epochs" : int(g_epochs),
        "c_epochs" : int(c_epochs),
        "g_batch" : int(g_batch),
        "g_input_size" : int(g_input_size),
        "c_true_batch" : int(c_true_batch),
        "c_false_batch" : int(c_false_batch),
        "c_randfalse_batch" : int(c_randfalse_batch),
        "note_distance_basis" : int(note_distance_basis),
        "next_from_slider_end" : bool(next_from_slider_end),
        "max_ticks_for_ds" : int(max_ticks_for_ds),
        "box_loss_border" : float(box_loss_border),
        "box_loss_value" : float(box_loss_value),
        "box_loss_weight" : int(box_loss_weight)
    }
    
    step6_set_gan_params(GAN_PARAMS)
    osu_a, data = step6_run_all(flow_dataset_npz=model_params["flow_dataset"])
    
    return "Success"

def step3(stream_regularizer, slider_mirror):
    global osu_a, data
    # modding_params = model_params["modding"]
    modding_params = {
        "stream_regularizer": int(stream_regularizer),
        "slider_mirror": int(slider_mirror)
    }
    
    osu_a, data = step7_modding(osu_a, data, modding_params)
    
    # if select_model == "taiko":
        # taiko_hitsounds_params = step8_taiko_hitsounds_set_params(divisor=4, metronome_count=4)
        # hitsounds = step8_apply_taiko_hitsounds(osu_a, data, hs_dataset=model_params["hs_dataset"], params=taiko_hitsounds_params)
        # saved_osu_name = step8_save_osu_file(osu_a, data, hitsounds=hitsounds)
    # else:
    saved_osu_name = step8_save_osu_file_gui(osu_a, data)
    return saved_osu_name













# from act_newmap_prep import step4_read_new_map_gui
from mania_setup_colab import mania_load_pretrained_model
# rename functions +
from mania_act_rhythm_calc import mania_step5_load_model, mania_step5_load_npz, mania_step5_set_params, mania_step5_predict_notes, mania_step5_build_pattern, mania_modding, mania_merge_objects_each_key

def mania_step1(models, map_file, audio_file, note_density, hold_favor, divisor_favor, hold_max_ticks, hold_min_return, rotate_mode):
    global mania_notes_each_key
    # select_model = "lowkey" = models
    model_params = mania_load_pretrained_model(models)
    step4_read_new_map_gui(map_file.name, audio_file.name)
    # step4_read_new_map(uploaded_osu_name)
    

    model = mania_step5_load_model(model_file=model_params["rhythm_model"])
    npz = mania_step5_load_npz()
    # params = model_params["rhythm_param"]
    # Or set the parameters here...
    params = mania_step5_set_params(note_density=float(note_density), hold_favor=float(hold_favor), divisor_favor=[divisor_favor] * 4, hold_max_ticks=int(hold_max_ticks), hold_min_return=int(hold_min_return), rotate_mode=int(rotate_mode))
    
    predictions = mania_step5_predict_notes(model, npz, params)
    mania_notes_each_key = mania_step5_build_pattern(predictions, params, pattern_dataset=model_params["pattern_dataset"])
    return "{} notes predicted.".format(np.sum(predictions[0]))


from mania_act_final import step8_save_osu_mania_file_gui
def mania_step2(key_fix):
    global mania_notes_each_key
    # modding_params = model_params["modding"]
    modding_params = {
        "key_fix" : int(key_fix)
    }
    
    notes_each_key = mania_modding(mania_notes_each_key, modding_params)
    notes, key_count = mania_merge_objects_each_key(notes_each_key)
    saved_osu_name = step8_save_osu_mania_file_gui(notes, key_count)
    return saved_osu_name












def get_timed_osu_file(mode, audio_file, artist, title, beatmap_creator, difficulty_name, hp, cs, ar, od, sv, sltira, input_filename = "assets/my_template.osu", output_filename = "timing.osu"):
    with open(input_filename) as osu_file:
        osu_text = osu_file.read()

    if mode == "osu":
        game_mode = 0
    if mode == "taiko":
        game_mode = 1
    if mode == "catch":
        game_mode = 2
    if mode == "mania":
        game_mode = 3

    if beatmap_creator == "":
        beatmap_creator = "osumapper"

    # rdr = id3.Reader(music_path)
    # artist = rdr.get_value("performer")
    # if artist is None:
    #     artist = "unknown"
    # title = rdr.get_value("title")
    # if title is None:
    #     title = re.sub("\.[^\.]*$", "", os.path.basename(music_path))


    bpm, offset = get_timing(audio_file.name)
    osu_text = re.sub("{game_mode}", str(game_mode), osu_text)

    osu_text = re.sub("{title}", title, osu_text)
    osu_text = re.sub("{artist}", artist, osu_text)
    osu_text = re.sub("{creator}", beatmap_creator, osu_text)
    osu_text = re.sub("{version}", difficulty_name, osu_text)

    osu_text = re.sub("{hp_drain}", f"{hp}", osu_text)
    osu_text = re.sub("{circle_size}", f"{cs}", osu_text)
    osu_text = re.sub("{overall_difficulty}", f"{od}", osu_text)
    osu_text = re.sub("{approach_rate}", f"{ar}", osu_text)

    osu_text = re.sub("{slider_velocity}", f"{sv}", osu_text)
    osu_text = re.sub("{slider_tick_rate}", f"{sltira}", osu_text)

    osu_text = re.sub("{offset}", f"{int(offset)}", osu_text)
    osu_text = re.sub("{tickLength}", f"{60000 / bpm}", osu_text)
    
    with open(output_filename, 'w', encoding="utf8") as osu_output:
        osu_output.write(osu_text)

    return output_filename

def get_timed_osu_file_mania(audio_file, artist, title, beatmap_creator, difficulty_name, hp, cs, od, input_filename = "assets/my_template.osu", output_filename = "timing.osu"):
    with open(input_filename) as osu_file:
        osu_text = osu_file.read()

    game_mode = 3
    ar = 5
    sv = 1.40
    sltira = 1

    if beatmap_creator == "":
        beatmap_creator = "osumapper"

    # rdr = id3.Reader(music_path)
    # artist = rdr.get_value("performer")
    # if artist is None:
    #     artist = "unknown"
    # title = rdr.get_value("title")
    # if title is None:
    #     title = re.sub("\.[^\.]*$", "", os.path.basename(music_path))


    bpm, offset = get_timing(audio_file.name)
    osu_text = re.sub("{game_mode}", str(game_mode), osu_text)

    osu_text = re.sub("{title}", title, osu_text)
    osu_text = re.sub("{artist}", artist, osu_text)
    osu_text = re.sub("{creator}", beatmap_creator, osu_text)
    osu_text = re.sub("{version}", difficulty_name, osu_text)

    osu_text = re.sub("{hp_drain}", f"{hp}", osu_text)
    osu_text = re.sub("{circle_size}", f"{cs}", osu_text)
    osu_text = re.sub("{overall_difficulty}", f"{od}", osu_text)
    osu_text = re.sub("{approach_rate}", f"{ar}", osu_text)

    osu_text = re.sub("{slider_velocity}", f"{sv}", osu_text)
    osu_text = re.sub("{slider_tick_rate}", f"{sltira}", osu_text)

    osu_text = re.sub("{offset}", f"{int(offset)}", osu_text)
    osu_text = re.sub("{tickLength}", f"{60000 / bpm}", osu_text)
    
    with open(output_filename, 'w', encoding="utf8") as osu_output:
        osu_output.write(osu_text)

    return output_filename

with gr.Blocks(title="WebUI") as app:
    gr.Markdown(value="sota Sota Fujimori music(☆>5.0) vtuber(☆4.0-5.3) inst(☆3.5-6.5) tvsize(☆3.5-5.0 BPM140-190) hard(☆<3.5 BPM140-190) normal(☆<2.7 BPM140-190) lowbpm(☆3-4.5 BPM<140) taiko experimental(☆3-6) catch experimental(☆3-6) mytf8star(☆8)")
    with gr.Tabs():
        with gr.TabItem("osu std"):
            models = gr.Radio(label="models", choices=["default", "sota", "vtuber", "inst", "tvsize", "hard", "normal", "lowbpm", "mytf8star"], value="default", interactive=True)#, "taiko", "catch"
            with gr.Row():
                map_file = gr.File(label="Drop your Map file here")
                audio_file = gr.File(label="Drop your Audio file here")
                with gr.Column():
                    dist_multiplier = gr.Slider(minimum=0, maximum=1000, value=1, label="dist_multiplier", interactive=True)
                    note_density = gr.Slider(minimum=0, maximum=1, value=0.32, label="note_density", interactive=True)
                    slider_favor = gr.Slider(minimum=-1.1, maximum=1.1, value=0, label="slider_favor", interactive=True)
                    divisor_favor = gr.Slider(minimum=-1, maximum=1, value=0, label="divisor_favor", interactive=True)
                    slider_max_ticks = gr.Slider(minimum=0, maximum=1000, value=8, label="slider_max_ticks", interactive=True)
                butstep1 = gr.Button("step1", variant="primary")
            with gr.Row():
                output1 = gr.Textbox(label="Output information")
            butstep1.click(step1, [models, map_file, audio_file, dist_multiplier, note_density, slider_favor, divisor_favor, slider_max_ticks], [output1], api_name="convert")
            with gr.Row():
                good_epoch = gr.Slider(minimum=0, maximum=1000, value=6, label="good_epoch", interactive=True)
                max_epoch = gr.Slider(minimum=0, maximum=1000, value=25, label="max_epoch", interactive=True)
                note_distance_basis = gr.Slider(minimum=0, maximum=1000, value=200, step=10, label="note_distance_basis", interactive=True)
                max_ticks_for_ds = gr.Slider(minimum=0, maximum=1000, value=2, label="max_ticks_for_ds", interactive=True)
                next_from_slider_end = gr.Radio(label="next_from_slider_end", choices=["True", "False"], value="False", interactive=True)
                box_loss_border = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.01, label="box_loss_border", interactive=True)
                box_loss_value = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.04, label="box_loss_value", interactive=True)
                box_loss_weight = gr.Slider(minimum=0, maximum=10, value=1, label="box_loss_weight", interactive=True)
                with gr.Accordion('GAN', open=False):
                    with gr.Column():
                        g_epochs = gr.Slider(minimum=0, maximum=1000, value=7, label="g_epochs", interactive=True)
                        g_batch = gr.Slider(minimum=0, maximum=1000, value=50, label="g_batch", interactive=True)
                        g_input_size = gr.Slider(minimum=0, maximum=1000, value=50, label="g_input_size", interactive=True)
                    with gr.Column():
                        c_epochs = gr.Slider(minimum=0, maximum=1000, value=3, label="c_epochs", interactive=True)
                        c_true_batch = gr.Slider(minimum=0, maximum=1000, value=50, label="c_true_batch", interactive=True)
                        c_false_batch = gr.Slider(minimum=0, maximum=1000, value=5, label="c_false_batch", interactive=True)
                        c_randfalse_batch = gr.Slider(minimum=0, maximum=1000, value=5, label="c_randfalse_batch", interactive=True)
                butstep2 = gr.Button("step2", variant="primary")
            with gr.Row():
                output2 = gr.Textbox(label="Output information", scale=0)
                # gr.Image(value="graph0.png")
            butstep2.click(step2, [good_epoch, max_epoch, note_distance_basis, max_ticks_for_ds, next_from_slider_end, box_loss_border, box_loss_value, box_loss_weight, g_epochs, g_batch, g_input_size, c_epochs, c_true_batch, c_false_batch, c_randfalse_batch], [output2], api_name="gan")
            with gr.Row():
                with gr.Column():
                    stream_regularizer = gr.Slider(minimum=0, maximum=4, step=1, value=1, label="stream_regularizer", interactive=True)
                    slider_mirror = gr.Slider(minimum=0, maximum=1, step=1, value=1, label="slider_mirror", interactive=True)
                butstep3 = gr.Button("step3", variant="primary")
                file_out = gr.File(interactive=False, label="map file output")
                butstep3.click(step3, [stream_regularizer, slider_mirror], [file_out], api_name="mod")
            # with gr.Row():
            #     butstep3 = gr.Button("clean up", variant="primary")
        # with gr.TabItem("taiko"):


        with gr.TabItem("mania"):
            models = gr.Radio(label="models", choices=["lowkey", "highkey"], value="lowkey", interactive=True)# no model for "default"
            with gr.Row():
                map_file = gr.File(label="Drop your Map file here")
                audio_file = gr.File(label="Drop your Audio file here")
                with gr.Column():
                    note_density = gr.Slider(minimum=0, maximum=1, value=0.55, label="note_density", interactive=True)
                    hold_favor = gr.Slider(minimum=-1, maximum=1, value=0.12, label="hold_favor", interactive=True)
                    divisor_favor = gr.Slider(minimum=-1, maximum=1, value=0, label="divisor_favor", interactive=True)
                    hold_max_ticks = gr.Slider(minimum=1, maximum=1000, value=8, label="hold_max_ticks", interactive=True)
                    hold_min_return = gr.Slider(minimum=1, maximum=1000, value=5, label="hold_min_return", interactive=True)
                    rotate_mode = gr.Slider(minimum=0, maximum=4, step=1, value=4, label="rotate_mode", interactive=True)
                mania_butstep1 = gr.Button("step1", variant="primary")
            with gr.Row():
                output1 = gr.Textbox(label="Output information")
            mania_butstep1.click(mania_step1, [models, map_file, audio_file, note_density, hold_favor, divisor_favor, hold_max_ticks, hold_min_return, rotate_mode], [output1], api_name="mania_convert")
            with gr.Row():
                with gr.Column():
                    key_fix = gr.Slider(minimum=0, maximum=3, step=1, value=1, label="key_fix", interactive=True)
                mania_butstep2 = gr.Button("step2", variant="primary")
                file_out = gr.File(interactive=False, label="map file output")
                mania_butstep2.click(mania_step2, [key_fix], [file_out], api_name="mania_mod")

        
        # with gr.TabItem("catch"):
        with gr.TabItem("empty .osu file maker"):
            # with gr.Row():
            with gr.TabItem("osu"):
                mode = gr.Radio(label="mode", choices=["osu", "taiko", "catch"], value="osu", interactive=True)
                audio_file = gr.File(label="Drop your Audio file here")
                with gr.Row():
                    with gr.Column():
                        artist = gr.Textbox(label="Artist", value="", interactive=True)
                        title = gr.Textbox(label="Title", value="", interactive=True)
                        beatmap_creator = gr.Textbox(label="Beatmap Creator", value="", interactive=True)
                        difficulty_name = gr.Textbox(label="Difficulty Name", value="", interactive=True)
                    with gr.Column():
                        hp = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="HP Drain Rate", interactive=True)
                        cs = gr.Slider(minimum=2, maximum=7, value=5, step=0.1, label="Circle Size", interactive=True)
                        ar = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="Approach Rate", interactive=True)
                        od = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="Overall Difficulty", interactive=True)
                        sv = gr.Slider(minimum=0.4, maximum=3.60, value=1.40, step=0.1, label="Slider Velocity", interactive=True)
                        sltira = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Slider Tick Rate", interactive=True)
                butmake = gr.Button("make .osu file", variant="primary")
                file_out2 = gr.File(interactive=False, label="map file output")
                butmake.click(get_timed_osu_file, [mode, audio_file, artist, title, beatmap_creator, difficulty_name, hp, cs, ar, od, sv, sltira], [file_out2], api_name="make")
            with gr.TabItem("mania"):
                audio_file2 = gr.File(label="Drop your Audio file here")
                with gr.Row():
                    with gr.Column():
                        artist2 = gr.Textbox(label="Artist", value="", interactive=True)
                        title2 = gr.Textbox(label="Title", value="", interactive=True)
                        beatmap_creator2 = gr.Textbox(label="Beatmap Creator", value="", interactive=True)
                        difficulty_name2 = gr.Textbox(label="Difficulty Name", value="", interactive=True)
                    with gr.Column():
                        hp2 = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="HP Drain Rate", interactive=True)
                        key_count = gr.Slider(minimum=1, maximum=9, value=5, step=1, label="Key Count", interactive=True)
                        od2 = gr.Slider(minimum=0, maximum=10, value=5, step=0.1, label="Overall Difficulty", interactive=True)
                butmake2 = gr.Button("make .osu file", variant="primary")
                file_out3 = gr.File(interactive=False, label="map file output")
                butmake2.click(get_timed_osu_file_mania, [audio_file2, artist2, title2, beatmap_creator2, difficulty_name2, hp2, key_count, od2], [file_out3], api_name="make2")
        

    if cmd_opts.colab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=True,
            server_port=7865,
            quiet=True,
        )