import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.optimize import minimize

data = """SHARP0 = (24,1,9.4576845)
PULL0 = (24,1,10.4002303)
SHARP1 = (24,5,12.5206635)
PULL1 = (24,5,11.7930717)
SHARP2 = (24,10,16.2006062)
PULL2 = (24,10,11.3718273)
SHARP3 = (24,15,22.9821362)
PULL3 = (24,15,12.5739862)
SHARP4 = (24,20,30.2705635)
PULL4 = (24,20,9.4699517)
SHARP5 = (18,1,6.9011094)
PULL5 = (18,1,7.2723574)
SHARP6 = (18,5,8.1836534)
PULL6 = (18,5,7.1257585)
SHARP7 = (18,10,11.3727328)
PULL7 = (18,10,7.1117468)
SHARP8 = (18,15,16.9265305)
PULL8 = (18,15,8.8708287)
SHARP9 = (18,20,22.8926633)
PULL9 = (18,20,8.495551)
SHARP10 = (12,1,4.3565068)
PULL10 = (12,1,5.1252367)
SHARP11 = (12,5,6.0172134)
PULL11 = (12,5,5.5167734)
SHARP12 = (12,10,7.5904766)
PULL12 = (12,10,5.8887988)
SHARP13 = (12,15,11.0382068)
PULL13 = (12,15,5.7805567)
SHARP14 = (12,20,15.9193988)
PULL14 = (12,20,5.790875)
SHARP15 = (6,1,2.2728882)
PULL15 = (6,1,2.0499862)
SHARP16 = (6,5,2.6400067)
PULL16 = (6,5,2.6622246)
SHARP17 = (6,10,4.1085335)
PULL17 = (6,10,2.6747647)
SHARP18 = (6,15,5.2891621)
PULL18 = (6,15,2.0516185)
SHARP19 = (6,20,7.9248217)
PULL19 = (6,20,2.7766207)"""

points = data.split("\n")
points = [point.split(" = ")[1] for point in points]
points = [point.replace("(", "").replace(")", "").split(",") for point in points]
points = sorted(sorted([(int(point[0]), int(point[1]), float(point[2])) for point in points], key=lambda x: x[0]), key=lambda x: x[1])

sharp_points = [point for i, point in enumerate(points) if i % 2 == 0]
pull_points = [point for i, point in enumerate(points) if i % 2 != 0]


x_p = np.linspace(0, max([x for x,y,z in points]), 100)
y_p = np.linspace(0, max([y for x,y,z in points]), 100)
func1_factor = sharp_points[-1][2] / (sharp_points[-1][0] + sharp_points[-1][1]) ** 2.2
print(func1_factor)
x_p, y_p = np.meshgrid(x_p, y_p)

def func_1(x, y):
    return func1_factor* (x+y)**2.2

def func_2(x, y, a, b):
    return (func_1(x,y)**(1/2.2)) + a * x - b * y

def func_2_err(params):
    a, b = params
    return sum((func_2(x, y, a, b) - z) ** 2 for x, y, z in pull_points)

inital_guess = [1.0, 1.0]

result = minimize(func_2_err, inital_guess)
a, b = result.x
print(f"a: {a}, b: {b}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax_rot_x = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_rot_y = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_x = Slider(ax_rot_x, "Rotate X", 0, 360, valinit=30)
slider_y = Slider(ax_rot_y, "Rotate Y", 0, 360, valinit=110)

def update(*_):
    ax.view_init(elev=slider_x.val, azim=slider_y.val)
    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_y.on_changed(update)

# Unpack points into x, y, z lists
# Plot points
ax.scatter(*zip(*sharp_points), color='red', marker='o')
ax.scatter(*zip(*pull_points), color='blue', marker='^')
ax.plot_surface(x_p, y_p, func_1(x_p, y_p), color='red', alpha=0.5)
ax.plot_surface(x_p, y_p, func_2(x_p, y_p, a, b), color='blue', alpha=0.5)

## show a and b
ax.text2D(0.05, 0.95, f"a: {a:.5f}, b: {b:.5f}", transform=ax.transAxes)


# Labels
ax.set_xlabel("Image count")
ax.set_ylabel("Radius")
ax.set_zlabel("Time (s)")
update()
plt.show()
