Thermal Image Processing Script

What this script does:

1.Converts thermal images to colorized versions using colormaps
2.Enhances image resolution using AI (EDSR model) or traditional upscaling
3.Organizes results into easy-to-find folders

How to install:
1. Install Python python.org/downloads
Check "Add Python to PATH" during installation
2. Check if installed: Press Win+R, type cmd, press enter, then type python --version
3. Install Required Libraries, open cmd again and paste these commands one by one:
pip install numpy
pip install opencv-contrib-python-headless
pip install matplotlib
pip install tqdm
pip install infiray-irg 

How to use:
Open a config.json file in the same folder as the script, then:
1. input_dir: type path to folder with your .irg files
2. output_dir: type path where processed images will be saved
3. model_path: type path to EDSR model (in the same folder as the script, "EDSR_x4.pb")
4. colormap_name: type the color style (try: "viridis", "plasma", "magma", or run all_colormaps.py to see full list of possible colormaps
5. scale_factor: 4 by default, but can be changed to 2 or 3, but required correct EDSR version, which you can find via internet

Keep in mind: use only Latin characters in path names, Cyrilic will cause a critical error! Also do not use backslashes in path names, forward slashes required!

Common errors and solutions:
1.ModuleNotFoundError: No module named 'infiray_irg'
Fix: Reinstall the library:

Type in cmd:
pip install --force-reinstall infiray-irg

2.Model Loading Errors

Model error: Cannot open "EDSR_x4.pb"
Invalid scale_factor

Confirm model path in config.json is correct
Ensure scale_factor matches your model (4 for EDSR_x4.pb)
Redownload the model if corrupted

3.JSONDecodeError in config.json

Remove commas after last items in .json

4.No .irg files found

Check:

input_dir points to correct folder
Thermal files have .irg extension


Support:

Found another issue? Provide these details:

Full error message from Command Prompt
Your config.json content (hide personal paths)
Example input file if possible

Email: znatyadeveloper14@gmail.com

Also you may need .irg graph constructor by jaseg (third-party developer):
https://github.com/jaseg/infiray_irg/blob/main/README.md