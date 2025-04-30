import os
import subprocess


class StructureDistance:
    @staticmethod
    def process_structure_distance(input_dir, output_dir, executable_path):
        for filename in os.listdir(input_dir):
            if filename.endswith(".pdb"):
                input_file = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                output_json = os.path.join(output_dir, f"{base_name}.json")
                output_pdb = os.path.join(output_dir, f"{base_name}.pdb")
                subprocess.run([
                    executable_path,
                    "-i", input_file,
                    "-j", output_json,
                    "-p", output_pdb,
                    "-d", "8.0"
                ])

