extends Node2D

@export var ar_scale: float = 3.0
@export var hands: Array[Line2D] = []

@export var left_fingers_parent: Node2D
@export var right_fingers_parent: Node2D

var left_fingers: Dictionary[String, Line2D] = {}
var right_fingers: Dictionary[String, Line2D] = {}

@export var torso: Polygon2D

@export var rend: ColorRect

var process_io: FileAccess
var process = null
var thread: Thread

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

	# process = OS.execute_with_pipe("python", ['./resources/py_scripts/face.py'])
	process = OS.execute_with_pipe("python", ['./resources/py_scripts/face_rewrite.py'])
	process_io = process['stdio']
	thread = Thread.new()
	thread.start(read_process_output)

func _exit_tree() -> void:
	process_io.store_line("q")
	thread.wait_to_finish()

func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("quit"):
		print("Tried quitting!")
		var res = process_io.store_line("q")
		print("RES: ", res)

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
			print("head: ", vec)
		elif line.begins_with("hand "):
			line = line.substr(4, -1)
			# print("CHARATCER:l ", line1)
			var hand_idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var p_flat = line.split(", ")
			var points = []
			for i in range(0, p_flat.size(), 2):
				points.append(p_flat[i] + ", " + p_flat[i + 1])
			for p_idx in points.size():
				var p = points[p_idx]
				var s = p.substr(1, p.length() - 2).split(", ")
				var v = Vector2(float(s[0]), float(s[1]))
				update_point_pos.call_deferred(hand_idx, p_idx, v)
		elif line.begins_with("shoulder "):
			line = line.substr(8, -1)
			# print("CHARATCER:l ", line1)
			var idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var s = line.split(", ")
			var v = Vector2(float(s[0]), float(s[1]))
			update_shoulder.call_deferred(idx, v)
			# print("shoulder ", idx, ": ", v)
		elif line.begins_with("hip "):
			line = line.substr(3, -1)
			# print("CHARATCER:l ", line1)
			var idx := int(line[1])
			line = line.substr(5, line.length() - 6)
			var s = line.split(", ")
			var v = Vector2(float(s[0]), float(s[1]))
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
				for i in range(0, p_flat.size(), 2):
					points.append(p_flat[i] + ", " + p_flat[i + 1])
				# print(finger, " ", hand_idx, ": ", points)
				for p_idx in points.size():
					var p = points[p_idx]
					var s = p.substr(1, p.length() - 2).split(", ")
					var v = Vector2(float(s[0]), float(s[1]))
					# update_point_pos.call_deferred(hand_idx, p_idx, v)
					update_finger.call_deferred(hand_idx, finger, p_idx, v)

func update_finger(hand_idx: int, finger: String, p_idx: int, v: Vector2) -> void:
	if hand_idx == 0:
		left_fingers[finger].set_point_position(p_idx, v)
	else:
		right_fingers[finger].set_point_position(p_idx, v)


func update_shoulder(idx: int, v: Vector2) -> void:
	var poly = torso.polygon
	poly.set(0 + idx, v)
	torso.polygon = poly

func update_hip(idx: int, v: Vector2) -> void:
	var poly = torso.polygon
	poly.set(3 - idx, v)
	torso.polygon = poly

func update_point_pos(hand_idx: int, p_idx: int, v: Vector2) -> void:
	hands[hand_idx].set_point_position(p_idx, v)

func update_angles(angle: Vector3) -> void:
	(rend.material as ShaderMaterial).set_shader_parameter("angles", angle)

func update_positions(pos: Vector2) -> void:
	rend.get_parent().position = pos

func update_scale(ar: float) -> void:
	rend.get_parent().scale.y = ar * ar_scale
