extends Camera2D

## Movement and zoom properties (exported for easy adjustment in the editor)
@export var move_speed: float = 500.0
@export var zoom_sensitivity: float = 0.1
@export var min_zoom: float = 0.1
@export var max_zoom: float = 3.0

# Current zoom level
var _current_zoom: float = 1.0

func _ready():
	# Set initial zoom
	_current_zoom = zoom.x
	
func _process(delta):
	# Handle camera movement
	var input_dir = Vector2.ZERO
	
	if Input.is_action_pressed("move_left"):
		input_dir.x = -1
	if Input.is_action_pressed("move_right"):
		input_dir.x = 1
	if Input.is_action_pressed("move_up"):
		input_dir.y = -1
	if Input.is_action_pressed("move_down"):
		input_dir.y = 1
	
	# Normalize to prevent faster diagonal movement
	if input_dir.length() > 0:
		input_dir = input_dir.normalized()
	
	# Apply movement
	position += input_dir * move_speed * delta / _current_zoom

func _input(event):
	# Handle zoom with mouse scroll
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			# Zoom in
			_zoom_camera(-zoom_sensitivity)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			# Zoom out
			_zoom_camera(zoom_sensitivity)

func _zoom_camera(zoom_factor):
	# Calculate new zoom level
	_current_zoom = clamp(_current_zoom + zoom_factor, min_zoom, max_zoom)
	
	# Apply zoom
	var new_zoom = Vector2(_current_zoom, _current_zoom)
	zoom = new_zoom
