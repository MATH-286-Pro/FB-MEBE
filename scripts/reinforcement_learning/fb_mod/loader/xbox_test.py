app_cfg = {
    "headless":       False,
    "device":         "cuda",
    "enable_cameras": False,
}

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(app_cfg).app
############ Launch Isaaclab APP #############

from xbox import XboxGamepad

class TEST:
    def __init__(self):
        self.xbox = XboxGamepad()
        print("Xbox Gamepad initialized!")
        print(self.xbox)
        print("Move sticks and press buttons to see values change...")
        print("=" * 80)
    
    def __call__(self):
        while simulation_app.is_running():
            # IMPORTANT: Update the simulation to process events
            simulation_app.update()
            
            values = self.xbox.advance()
            
            # Only print if any input is non-zero/non-false
            has_input = False
            output_lines = []
            
            # Check sticks
            for key in ['LX', 'LY', 'RX', 'RY']:
                if abs(values[key]) > 0.01:
                    output_lines.append(f"{key}: {values[key]:+.3f}")
                    has_input = True
            
            # Check triggers
            for key in ['LT', 'RT']:
                if values[key] > 0.01:
                    output_lines.append(f"{key}: {values[key]:.3f}")
                    has_input = True
            
            # Check buttons
            for key in ['A', 'B', 'X', 'Y', 'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT']:
                if values[key]:
                    output_lines.append(f"{key}: PRESSED")
                    has_input = True
            
            if has_input:
                print(" | ".join(output_lines))
            else:
                # Print a heartbeat every second to show it's running
                import sys
                sys.stdout.write(".")
                sys.stdout.flush()

if __name__ == "__main__":
    test = TEST()
    test()