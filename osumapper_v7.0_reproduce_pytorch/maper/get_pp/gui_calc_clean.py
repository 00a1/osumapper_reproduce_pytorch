import json
import math
from get_pp.diff_calc_m_clean import calc_stars
import get_pp.pp_calc_m as pp_calc_m

def get_all_slider_types():
    return [
        {"index": 0, "name": "linear", "type": "L", "vecLength": 1, "repeats": 1, "angle": 0, "points": [[1, 0]]},
        {"index": 1, "name": "arc-ccw", "type": "P", "vecLength": 0.97, "repeats": 1, "angle": -0.40703540572409336, "points": [[0.5, 0.1], [0.97, 0]]},
        {"index": 2, "name": "arc-cw", "type": "P", "vecLength": 0.97, "repeats": 1, "angle": 0.40703540572409336, "points": [[0.5, -0.1], [0.97, 0]]},
        {"index": 3, "name": "angle-ccw", "type": "B", "vecLength": 0.97, "repeats": 1, "angle": -0.20131710837464062, "points": [[0.48, 0.1], [0.48, 0.1], [0.97, 0]]},
        {"index": 4, "name": "angle-cw", "type": "B", "vecLength": 0.97, "repeats": 1, "angle": 0.20131710837464062, "points": [[0.48, -0.1], [0.48, -0.1], [0.97, 0]]},
        {"index": 5, "name": "wave-cw", "type": "B", "vecLength": 0.97, "repeats": 1, "angle": -0.46457807316944644, "points": [[0.38, -0.2], [0.58, 0.2], [0.97, 0]]},
        {"index": 6, "name": "wave-ccw", "type": "B", "vecLength": 0.97, "repeats": 1, "angle": 0.46457807316944644, "points": [[0.38, 0.2], [0.58, -0.2], [0.97, 0]]},
        {"index": 7, "name": "halfcircle-cw", "type": "P", "vecLength": 0.64, "repeats": 1, "angle": 1.5542036732051032, "points": [[0.32, -0.32], [0.64, 0]]},
        {"index": 8, "name": "halfcircle-ccw", "type": "P", "vecLength": 0.64, "repeats": 1, "angle": -1.5542036732051032, "points": [[0.32, 0.32], [0.64, 0]]},
        {"index": 9, "name": "haneru-cw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": 0, "points": [[0.24, -0.08], [0.44, -0.04], [0.64, 0.1], [0.64, 0.1], [0.76, 0], [0.94, 0]]},
        {"index": 10, "name": "haneru-ccw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": 0, "points": [[0.24, 0.08], [0.44, 0.04], [0.64, -0.1], [0.64, -0.1], [0.76, 0], [0.94, 0]]},
        {"index": 11, "name": "elbow-cw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": 0.23783592745745077, "points": [[0.28, -0.16], [0.28, -0.16], [0.94, 0]]},
        {"index": 12, "name": "elbow-ccw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": -0.23783592745745077, "points": [[0.28, 0.16], [0.28, 0.16], [0.94, 0]]},
        {"index": 13, "name": "ankle-cw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": 0.5191461142465229, "points": [[0.66, -0.16], [0.66, -0.16], [0.94, 0]]},
        {"index": 14, "name": "ankle-ccw", "type": "B", "vecLength": 0.94, "repeats": 1, "angle": -0.5191461142465229, "points": [[0.66, 0.16], [0.66, 0.16], [0.94, 0.0]]},
        {"index": 15, "name": "bolt-cw", "type": "B", "vecLength": 0.96, "repeats": 1, "angle": -0.16514867741462683, "points": [[0.34, -0.06], [0.34, -0.06], [0.6, 0.06], [0.6, 0.06], [0.96, 0.0]]},
        {"index": 16, "name": "bolt-ccw", "type": "B", "vecLength": 0.96, "repeats": 1, "angle": 0.16514867741462683, "points": [[0.34, 0.06], [0.34, 0.06], [0.6, -0.06], [0.6, -0.06], [0.96, 0.0]]},
        {"index": 17, "name": "linear-reverse", "type": "L", "vecLength": 0, "repeats": 2, "angle": 3.141592653589793, "points": [[0.0, 0.0], [0.5, 0.0]]}
    ]

class mods:
    def __init__(self):
        self.nomod = 0,
        self.nf = 0
        self.ez = 0
        self.hd = 0
        self.hr = 0
        self.dt = 0
        self.ht = 0
        self.nc = 0
        self.fl = 0
        self.so = 0
        self.td = 0
        self.speed_changing = self.dt | self.ht | self.nc
        self.map_changing = self.hr | self.ez | self.speed_changing
    def update(self):
        self.speed_changing = self.dt | self.ht | self.nc
        self.map_changing = self.hr | self.ez | self.speed_changing

class slider_data:
    def __init__(self, length):#, s_type, points, repeats,
        # self.s_type = s_type
        # self.points = points
        # self.repeats = repeats
        self.length = length

class hit_object:
    def __init__(self, pos, time, h_type, end_time, slider):
        self.pos = pos
        self.time = time
        self.h_type = h_type
        self.end_time = end_time
        self.slider = slider

class timing_point:
    def __init__(self, time, ms_per_beat, inherit):
        self.time = time
        self.ms_per_beat = ms_per_beat
        self.inherit = inherit

def convert_to_osu_obj(obj_array, data, diff, timing_points, hitsounds=None):
    """
    Converts map data from python format to json format.
    """
    tick_rate = 1
    speed = 1
    
    # Combo
    num_circles = 0
    num_sliders = 0
    num_spinners = 0
    max_combo = 0
    num_objects = 0
    objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, slider_types, slider_length_base = data

    if hitsounds is None:
        hitsounds = [0] * len(obj_array)

    output = []
    objects = []
    ho_num = 0
    for i, obj in enumerate(obj_array):
        slider_types_list = get_all_slider_types()
        pos = [obj[0], obj[1]]
        slider = 0
        end_time = 0
        time = timestamps[i]
        if not is_slider[i]: # is a circle does not consider spinner for now.
            num_circles += 1
            h_type = 1
        # elif is_spinner:# add
        #     self.num_spinners += 1
        #     h_type = 3
        else:
            num_sliders += 1
            h_type = 2
            length = float(slider_length_base[i] * slider_ticks[i])

            time_p = timing_points[0]# if get timeing point breaks first use this
            st = slider_types_list[int(slider_types[i])]
            repeats = st["repeats"]# mostly 1.0

            # Get timing point
            for tp in timing_points:
                if float(tp.time) > float(time): #245 > slider time(1268) = 0 pass to next tp if next tp is 95701 > slider time(1268) = 1 use 95701 for time_p
                    break
                time_p = tp # time_p if full not just time
            
            # Begin to calculte the amount of ticks for max combo
            sv_mult = 1
            if time_p.inherit == "0" and float(tp.ms_per_beat) < 0:
                sv_mult = (-100.0 / float(time_p.ms_per_beat))
            px_per_beat = diff["SV"] * 100.0 * sv_mult
            num_beats = (length * repeats) / px_per_beat
            # duration = math.ceil(num_beats * float(parent.ms_per_beat))
            # end_time = float(time) + duration

            slider = slider_data(length)#sl_type,pos_s,repeats,

            # ticks = math.ceil((num_beats - 0.1) / repeats * tick_rate)
            # ticks -= 1
            # ticks *= repeats
            # ticks += repeats + 1
            # max_combo += ticks - 1

            #small
            ticks = repeats * math.ceil((num_beats - 0.1) / repeats * tick_rate) + repeats
            max_combo += ticks - 1

        num_objects += 1
        max_combo += 1
        objects.append(hit_object(pos, time, h_type, end_time, slider))
    return objects, diff, speed, num_circles, num_sliders, num_spinners, max_combo, num_objects

def reparseTimeSections(tsa, map_diff):
    output_list = []
    sliderBaseLength = 100
    for ts in tsa:
        if ts["isInherited"]:
            tl = -sliderBaseLength * map_diff["SV"] * (100 / ts["sliderLength"])
        else:
            tl = ts["tickLength"]

        sp = 1 if ts["sampleSet"] == "normal" else (3 if ts["sampleSet"] == "drum" else 2)

        #o += f"{ts['beginTime']},{tl},{ts['whiteLines']},{sp},{ts['customSet']},{ts['volume']},{int(not ts['isInherited'])},{int(ts['isKiai'])}\n"
        #output_list.append(f"{ts['beginTime']},{tl},{ts['whiteLines']},{sp},{ts['customSet']},{ts['volume']},{int(not ts['isInherited'])},{int(ts['isKiai'])}")
        output_list.append(timing_point(ts['beginTime'], tl, int(not ts['isInherited'])))

    return output_list

def calc_metrics(osu_map, data):
    with open("mapthis.json", encoding="utf-8") as map_json:
        map_dict = json.load(map_json)
        map_diff = map_dict["diff"]
        map_timing = map_dict["timing"]["ts"]
    
    timing_points = reparseTimeSections(map_timing, map_diff)
    
    objects, diff, speed, num_circles, num_sliders, num_spinners, max_combo, num_objects = convert_to_osu_obj(osu_map, data, map_diff, timing_points)
    calc_diff = calc_stars(objects, diff, speed, num_circles, num_sliders, num_spinners, max_combo, num_objects)

    # print(calc_diff[0])#aim s1.480214167207677 | c1.6653949015587537
    # print(calc_diff[1])#speed s0.8578887569443205 | c1.598202634853306
    # print(calc_diff[2])#stars s2.649265629283676 | c3.2971936697647832

    # add custom difficulty and misses
    c100 = 0
    c50 = 0
    misses = 0
    sv = 1
    acc = 0
    combo = 0
    combo = max_combo
    mod = mods()

    if acc == 0:
        pp = pp_calc_m.pp_calc(calc_diff[0], calc_diff[1], diff, speed, num_circles, num_sliders, num_spinners, max_combo, num_objects, misses, c100, c50, mod, combo, sv)
    else:
        pp = pp_calc_m.pp_calc_acc(calc_diff[0], calc_diff[1], diff, speed, num_circles, num_sliders, num_spinners, max_combo, num_objects, acc, mod, combo, misses, sv)
    
    # print(pp.pp)
    return round(calc_diff[2], 2), round(pp.pp, 2)