# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Xbox gamepad controller that returns all raw button/stick values."""

import numpy as np
import weakref
from collections.abc import Callable

import carb
import omni


class XboxGamepad:
    """Xbox gamepad controller that provides raw values for all buttons and sticks.
    
    This class provides access to all gamepad inputs without predefined mappings,
    allowing users to define their own command mappings.
    
    Available inputs:
        - LX, LY: Left stick X and Y axes
        - RX, RY: Right stick X and Y axes  
        - LT, RT: Left and right triggers
        - A, B, X, Y: Face buttons
        - DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT: D-pad buttons
    
    Example usage:
        >>> gamepad = XboxGamepad(dead_zone=0.01)
        >>> values = gamepad.advance()
        >>> print(f"Left stick X: {values['LX']}, Y: {values['LY']}")
        >>> print(f"A button: {values['A']}, X button: {values['X']}")
    """

    def __init__(self, dead_zone: float = 0.01):
        """Initialize the Xbox gamepad controller.
        
        Args:
            dead_zone: Magnitude of dead zone for analog inputs. Values below this
                threshold will be set to 0. Defaults to 0.01.
        """
        # turn off simulator gamepad control
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        
        # store inputs
        self.dead_zone = dead_zone
        
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        
        # Initialize raw value buffers
        # For sticks: (positive, negative) direction pairs
        self._raw_values = {
            'LX': np.zeros(2),      # Left stick X
            'LY': np.zeros(2),      # Left stick Y
            'RX': np.zeros(2),      # Right stick X
            'RY': np.zeros(2),      # Right stick Y
            'LT': 0.0,              # Left trigger
            'RT': 0.0,              # Right trigger
            'LB': False,            # Left bumper
            'RB': False,            # Right bumper
            'A': False,             # A button
            'B': False,             # B button
            'X': False,             # X button
            'Y': False,             # Y button
            'DPAD_UP': False,       # D-pad up
            'DPAD_DOWN': False,     # D-pad down
            'DPAD_LEFT': False,     # D-pad left
            'DPAD_RIGHT': False,    # D-pad right
        }
        
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        
    def __del__(self):
        """Unsubscribe from gamepad events."""
        if hasattr(self, '_gamepad_sub') and self._gamepad_sub is not None:
            self._input.unsubscribe_from_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None
    
    def __str__(self) -> str:
        """Returns: A string containing the information of the gamepad."""
        msg = f"Xbox Gamepad Controller: {self.__class__.__name__}\n"
        msg += f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft Stick: LX, LY\n"
        msg += "\tRight Stick: RX, RY\n"
        msg += "\tTriggers: LT, RT\n"
        msg += "\tBumpers: LB, RB\n"
        msg += "\tFace Buttons: A, B, X, Y\n"
        msg += "\tD-Pad: DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT\n"
        return msg
    
    """
    Operations
    """
    
    def reset(self):
        """Reset all input values to default state."""
        for key in self._raw_values:
            if isinstance(self._raw_values[key], np.ndarray):
                self._raw_values[key].fill(0.0)
            elif isinstance(self._raw_values[key], bool):
                self._raw_values[key] = False
            else:
                self._raw_values[key] = 0.0
    
    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        """Add additional callback functions for specific gamepad inputs.
        
        A list of available gamepad keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.
        
        Args:
            key: The gamepad button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func
    
    def advance(self) -> dict:
        """Get the current state of all gamepad inputs.
        
        Returns:
            A dictionary containing all gamepad input values:
            - 'LX', 'LY', 'RX', 'RY': Analog stick values in range [-1, 1]
            - 'LT', 'RT': Trigger values in range [0, 1]
            - 'LB', 'RB': Bumper states (bool)
            - 'A', 'B', 'X', 'Y': Button states (bool)
            - 'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT': D-pad states (bool)
        """
        result = {}
        
        # Resolve stick axes (handle bidirectional input)
        for key in ['LX', 'LY', 'RX', 'RY']:
            result[key] = self._resolve_axis(self._raw_values[key])
        
        # Copy trigger and button values directly
        result['LT'] = float(self._raw_values['LT'])
        result['RT'] = float(self._raw_values['RT'])
        result['LB'] = bool(self._raw_values['LB'])
        result['RB'] = bool(self._raw_values['RB'])
        result['A'] = bool(self._raw_values['A'])
        result['B'] = bool(self._raw_values['B'])
        result['X'] = bool(self._raw_values['X'])
        result['Y'] = bool(self._raw_values['Y'])
        result['DPAD_UP'] = bool(self._raw_values['DPAD_UP'])
        result['DPAD_DOWN'] = bool(self._raw_values['DPAD_DOWN'])
        result['DPAD_LEFT'] = bool(self._raw_values['DPAD_LEFT'])
        result['DPAD_RIGHT'] = bool(self._raw_values['DPAD_RIGHT'])
        
        return result
    
    """
    Internal helpers.
    """
    
    def _on_gamepad_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        """Subscriber callback when gamepad event occurs.
        
        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html
        """
        # Get current value and apply dead zone
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0
        
        # Left stick
        if event.input == carb.input.GamepadInput.LEFT_STICK_UP:
            self._raw_values['LY'][0] = cur_val  # positive Y
        elif event.input == carb.input.GamepadInput.LEFT_STICK_DOWN:
            self._raw_values['LY'][1] = cur_val  # negative Y
        elif event.input == carb.input.GamepadInput.LEFT_STICK_RIGHT:
            self._raw_values['LX'][0] = cur_val  # positive X
        elif event.input == carb.input.GamepadInput.LEFT_STICK_LEFT:
            self._raw_values['LX'][1] = cur_val  # negative X
        
        # Right stick
        elif event.input == carb.input.GamepadInput.RIGHT_STICK_UP:
            self._raw_values['RY'][0] = cur_val  # positive Y
        elif event.input == carb.input.GamepadInput.RIGHT_STICK_DOWN:
            self._raw_values['RY'][1] = cur_val  # negative Y
        elif event.input == carb.input.GamepadInput.RIGHT_STICK_RIGHT:
            self._raw_values['RX'][0] = cur_val  # positive X
        elif event.input == carb.input.GamepadInput.RIGHT_STICK_LEFT:
            self._raw_values['RX'][1] = cur_val  # negative X
        
        # Triggers
        elif event.input == carb.input.GamepadInput.LEFT_TRIGGER:
            self._raw_values['LT'] = cur_val
        elif event.input == carb.input.GamepadInput.RIGHT_TRIGGER:
            self._raw_values['RT'] = cur_val
        
        # Bumpers (Shoulder buttons)
        elif event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self._raw_values['LB'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self._raw_values['RB'] = cur_val > 0.5
        
        # Face buttons
        elif event.input == carb.input.GamepadInput.A:
            self._raw_values['A'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.B:
            self._raw_values['B'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.X:
            self._raw_values['X'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.Y:
            self._raw_values['Y'] = cur_val > 0.5
        
        # D-pad
        elif event.input == carb.input.GamepadInput.DPAD_UP:
            self._raw_values['DPAD_UP'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.DPAD_DOWN:
            self._raw_values['DPAD_DOWN'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.DPAD_LEFT:
            self._raw_values['DPAD_LEFT'] = cur_val > 0.5
        elif event.input == carb.input.GamepadInput.DPAD_RIGHT:
            self._raw_values['DPAD_RIGHT'] = cur_val > 0.5
        
        # Handle additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()
        
        return True
    
    def _resolve_axis(self, raw_axis: np.ndarray) -> float:
        """Resolve bidirectional axis input into a single value.
        
        Args:
            raw_axis: Array of shape (2,) containing [positive_value, negative_value]
        
        Returns:
            Resolved axis value in range [-1, 1]
        """
        # Determine sign: if negative value is larger, result should be negative
        if raw_axis[1] > raw_axis[0]:
            return -float(raw_axis[1])
        else:
            return float(raw_axis[0])
