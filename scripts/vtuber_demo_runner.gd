extends Node2D

@export var angle_scale := Vector3.ZERO
@export var ar_scale: float = 3.0
@export var hands: Array[Line2D] = []
@export var interpolation_speed: float = 4.0  # Added export variable for interpolation speed

@export var left_fingers_parent: Node2D
@export var right_fingers_parent: Node2D

var left_fingers: Dictionary[String, Line2D] = {}
var right_fingers: Dictionary[String, Line2D] = {}

@export var torso: Polygon2D

@export var rend: ColorRect

var process_io: FileAccess
var process = null
var thread: Thread

# Variables for interpolation
var shoulder_targets: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]
var shoulder_currents: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]
var hip_targets: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]
var hip_currents: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]

# Hand and finger point interpolation variables
var point_targets: Dictionary = {}
var point_currents: Dictionary = {}

const FINGERS: Array[StringName] = [
	&"thumb",
	&"index",
	&"middle",
	&"ring",
	&"pinky"
]

func _ready():
	for child in left_fingers_parent.get_children():
		left_fingers[child.name] = child
	
	for child in right_fingers_parent.get_children():
		right_fingers[child.name] = child
		
	# Initialize shoulder and hip positions from the torso polygon
	if torso and torso.polygon.size() >= 4:
		shoulder_currents[0] = Vector3(torso.polygon[0].x, torso.polygon[0].y, 0)
		shoulder_targets[0] = shoulder_currents[0]
		
		shoulder_currents[1] = Vector3(torso.polygon[1].x, torso.polygon[1].y, 0)
		shoulder_targets[1] = shoulder_currents[1]
		
		hip_currents[0] = Vector3(torso.polygon[2].x, torso.polygon[2].y, 0)
		hip_targets[0] = hip_currents[0]
		
		hip_currents[1] = Vector3(torso.polygon[3].x, torso.polygon[3].y, 0)
		hip_targets[1] = hip_currents[1]

	# process = OS.execute_with_pipe("python", ['./resources/py_scripts/face.py'])
	process = OS.execute_with_pipe("python", ['./resources/py_scripts/face_rewrite.py'])
	process_io = process['stdio']
	thread = Thread.new()
	thread.start(read_process_output)

func _exit_tree() -> void:
	process_io.store_line("q")
	thread.wait_to_finish()

func _process(delta: float) -> void:
	if Input.is_action_just_pressed("quit"):
		print("Tried quitting!")
		var res = process_io.store_line("q")
		print("RES: ", res)
	
	update_vars(delta)

func read_process_output():
	while process_io.is_open() and process_io.get_error() == OK:
		var line = process_io.get_line()
		# print(line)
		if line.begins_with("! "):
			line = line.substr(2, -1)
			var split = line.split(" ")
			var vec = Vector3(float(split[0]), float(split[1]) * 2.0, float(split[2]))
			update_angles.call_deferred(vec)
		elif line.begins_with("+ "):
			line = line.substr(2, -1)
			var split = line.split(" ")
			var vec = Vector2(float(split[0]), float(split[1]))
			update_positions.call_deferred(vec)
			# print("head: ", vec)
		elif line.begins_with("hand "):
			line = line.substr(4, -1)
			# print("CHARATCER:l ", line1)
			var hand_idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var p_flat = line.split(", ")
			var points = []
			for i in range(0, p_flat.size(), 3):
				points.append(p_flat[i] + ", " + p_flat[i + 1] + ", " + p_flat[i + 2])
			for p_idx in points.size():
				var p = points[p_idx]
				var s = p.substr(1, p.length() - 2).split(", ")
				var v = Vector3(float(s[0]), float(s[1]), float(s[2]))
				update_point_pos.call_deferred(hand_idx, p_idx, v)
		elif line.begins_with("shoulder "):
			line = line.substr(8, -1)
			# print("CHARATCER:l ", line1)
			var idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var s = line.split(", ")
			var v = Vector3(float(s[0]), float(s[1]), float(s[2]))
			update_shoulder.call_deferred(idx, v)
			# print("shoulder ", idx, ": ", v)
		elif line.begins_with("hip "):
			line = line.substr(3, -1)
			# print("CHARATCER:l ", line1)
			var idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var s = line.split(", ")
			var v = Vector3(float(s[0]), float(s[1]), float(s[2]))
			update_hip.call_deferred(idx, v)
			# print("hip ", idx, ": ", v)
		
		for finger in FINGERS:
			if line.begins_with(finger):
				line = line.substr(finger.length(), -1)
				# print("CHARATCER:l ", line1)
				var hand_idx := int(line[1])
				line = line.substr(5, line.length() - 6)
				var p_flat = line.split(", ")
				var points = []
				for i in range(0, p_flat.size(), 3):
					points.append(p_flat[i] + ", " + p_flat[i + 1] + ", " + p_flat[i + 2])
				# print(finger, " ", hand_idx, ": ", points)
				for p_idx in points.size():
					var p = points[p_idx]
					var s = p.substr(1, p.length() - 2).split(", ")
					var v = Vector3(float(s[0]), float(s[1]), float(s[2]))
					# update_point_pos.call_deferred(hand_idx, p_idx, v)
					update_finger.call_deferred(hand_idx, finger, p_idx, v)

func update_finger(hand_idx: int, finger: String, p_idx: int, v: Vector3) -> void:
	var key = str(hand_idx) + "_" + finger + "_" + str(p_idx)
	
	# Initialize dictionaries if needed
	if not point_targets.has(key):
		var current_pos = Vector3.ZERO
		
		# Get the current position from the correct finger line
		if hand_idx == 0 and left_fingers.has(finger):
			if p_idx < left_fingers[finger].get_point_count():
				var point = left_fingers[finger].get_point_position(p_idx)
				current_pos = Vector3(point.x, point.y, 0)
		elif hand_idx == 1 and right_fingers.has(finger):
			if p_idx < right_fingers[finger].get_point_count():
				var point = right_fingers[finger].get_point_position(p_idx)
				current_pos = Vector3(point.x, point.y, 0)
				
		point_targets[key] = current_pos
		point_currents[key] = current_pos
	
	# Set target position
	point_targets[key] = v

func update_shoulder(idx: int, v: Vector3) -> void:
	# Set target position for interpolation
	shoulder_targets[idx] = v

func update_hip(idx: int, v: Vector3) -> void:
	# Set target position for interpolation
	hip_targets[idx] = v

func update_point_pos(hand_idx: int, p_idx: int, v: Vector3) -> void:
	var key = "hand_" + str(hand_idx) + "_point_" + str(p_idx)
	
	# Initialize if needed
	if not point_targets.has(key):
		var current_pos = Vector3.ZERO
		
		# Get the current position if possible
		if hands.size() > hand_idx and p_idx < hands[hand_idx].get_point_count():
			var point = hands[hand_idx].get_point_position(p_idx)
			current_pos = Vector3(point.x, point.y, 0)
				
		point_targets[key] = current_pos
		point_currents[key] = current_pos
	
	# Set target position
	point_targets[key] = v

var target_angles := Vector3.ZERO
var curr_angle := Vector3.ZERO
func update_angles(angle: Vector3) -> void:
	target_angles = angle * angle_scale

var rend_target_pos := Vector2.ZERO
func update_positions(pos: Vector2) -> void:
	rend_target_pos = pos

var rend_target_scale := 0.0
func update_scale(ar: float) -> void:
	rend_target_scale = ar * ar_scale

func v3_v2(v: Vector3) -> Vector2:
	return Vector2(v.x, v.y)

func update_vars(delta: float) -> void:
	# Use exported interpolation_speed instead of hardcoded value
	var p := rend.get_parent()
	p.position = p.position.lerp(rend_target_pos, delta * interpolation_speed)
	# rend.get_parent().scale.y = lerpf(rend.get_parent().scale.y, rend_target_scale, delta * interpolation_speed)

	curr_angle = curr_angle.lerp(target_angles, delta * interpolation_speed)
	(rend.material as ShaderMaterial).set_shader_parameter("angles", curr_angle)

	# Interpolate shoulder and hip positions
	var poly_updated = false
	var poly = torso.polygon
	
	for i in range(shoulder_targets.size()):
		shoulder_currents[i] = shoulder_currents[i].lerp(shoulder_targets[i], delta * interpolation_speed)
		# Update the polygon directly (shoulders are at indices 0 and 1)
		poly.set(0 + i, v3_v2(shoulder_currents[i]))
		poly_updated = true
	
	for i in range(hip_targets.size()):
		hip_currents[i] = hip_currents[i].lerp(hip_targets[i], delta * interpolation_speed)
		# Update the polygon directly (hips are at indices 3 and 2)
		poly.set(3 - i, v3_v2(hip_currents[i]))
		poly_updated = true
	
	# Only update the polygon once after all changes
	if poly_updated:
		torso.polygon = poly

	# Interpolate all points in the dictionary (handles both hands and fingers)
	for key in point_targets.keys():
		# Interpolate current position toward target
		point_currents[key] = point_currents[key].lerp(point_targets[key], delta * interpolation_speed)
		
		# Apply the new position based on key prefix
		if key.begins_with("hand_"):
			# Extract hand index and point index from the key
			var parts = key.split("_")
			var hand_idx = int(parts[1])
			var point_idx = int(parts[3])
			
			if hands.size() > hand_idx and point_idx < hands[hand_idx].get_point_count():
				hands[hand_idx].set_point_position(point_idx, v3_v2(point_currents[key]))
		else:
			# This is a finger point
			var parts = key.split("_")
			var hand_idx = int(parts[0])
			var finger = parts[1]
			var point_idx = int(parts[2])
			
			if hand_idx == 0 and left_fingers.has(finger) and point_idx < left_fingers[finger].get_point_count():
				left_fingers[finger].set_point_position(point_idx, v3_v2(point_currents[key]))
			elif hand_idx == 1 and right_fingers.has(finger) and point_idx < right_fingers[finger].get_point_count():
				right_fingers[finger].set_point_position(point_idx, v3_v2(point_currents[key]))
