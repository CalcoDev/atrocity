extends Node2D

@export var ar_scale: float = 3.0
@export var hands: Array[Line2D] = []

@export var rend: ColorRect

var process_io: FileAccess
var process = null
var thread: Thread

func _ready():
	# process = OS.execute_with_pipe("python", ['./resources/py_scripts/sample.py'])
	process = OS.execute_with_pipe("python", ['./resources/py_scripts/face.py'])
	# process = OS.execute_with_pipe("python", ['./resources/py_scripts/face_angle_window.py'])
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
	var base_pos := Vector2.ZERO
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
			base_pos = vec
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
				update_point_pos.call_deferred(hand_idx, p_idx, v - base_pos)
		# elif line.begins_with("= "):
		# 	line = line.substr(2, -1)
		# 	var split = line.split(" ")
		# 	var aspect_raitio = float(split[0])
		# 	update_scale.call_deferred(aspect_raitio)

func update_point_pos(hand_idx: int, p_idx: int, v: Vector2) -> void:
	hands[hand_idx].set_point_position(p_idx, v)

func update_angles(angle: Vector3) -> void:
	(rend.material as ShaderMaterial).set_shader_parameter("angles", angle)

func update_positions(pos: Vector2) -> void:
	rend.position = pos

func update_scale(ar: float) -> void:
	rend.get_parent().scale.y = ar * ar_scale