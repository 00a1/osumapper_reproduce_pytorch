# Part 5 action script

from audio_tools import test_process_path, read_and_save_osu_tester_file
from os_tools import fix_path, test_node_modules

mapdata_path = "mapdata/"

def step4_read_new_map(file_path, divisor = 4):
    # fix the path
    fix_path()

    # Test paths and node
    test_process_path("node")

    # Test ffmpeg..?
    test_process_path("ffmpeg", "-version")

    # Test node modules
    test_node_modules()

    read_and_save_osu_tester_file(file_path.strip(), filename="mapthis", divisor=divisor)
