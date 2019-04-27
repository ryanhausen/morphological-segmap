import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

# setup matplotlib axes and params
fig, ax = plt.subplots(figsize=(5,5))

ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xticks([])
ax.set_yticks([])
# setup matplotlib axes and params

# drawing params
width, height = 6, 6
rect_color = "lightblue"
new_color = "lightorange"
edge_color = "black"
edge_width = 4

square_width = 2.3
square_edgewidth = 3
# drawing params

patches = []
patches_start_locs = []
# setup shapes
rect_1 = patch.Rectangle((2,2), width, height/3, fill=True, fc=rect_color)
patches_start_locs.append((2,2))
patches.append(rect_1)

rect_2 = patch.Rectangle((2,4), width, height/3, fill=True, fc=rect_color)
patches_start_locs.append((2,4))
patches.append(rect_2)

rect_3 = patch.Rectangle((2,6), width, height/3, fill=True, fc=rect_color)
patches_start_locs.append((2,6))
patches.append(rect_3)

# classification squares 
c_squares, c_squares_locs = [], []
classify_sq1 = patch.Rectangle((1.5, 0.85), square_width, square_width)
c_squares_locs.append((1.5,0.85))
c_squares.append(classify_sq1)

classify_sq2 = patch.Rectangle((1.5, 3.85), square_width, square_width)
c_squares_locs.append((1.5,3.85))
c_squares.append(classify_sq2)

classify_sq3 = patch.Rectangle((1.5, 6.85), square_width, square_width)
c_squares_locs.append((1.5,6.85))
c_squares.append(classify_sq3)

c_square_coll = PatchCollection(c_squares, facecolor="none", edgecolor=edge_color, linewidth=square_edgewidth, animated=True)


# new color squares
new_squares = []

new_square1 = patch.Rectangle((2, 7), 1.75, height/3)
new_squares.append(new_square1)

new_square2 = patch.Rectangle((2, 4), 1.75, height/3)
new_squares.append(new_square2)

new_square3 = patch.Rectangle((2, 1), 1.75, height/3)
new_squares.append(new_square3)

new_square_coll = PatchCollection(new_squares, facecolor="orange", animated=True)

# setup shapes

# set text
text = ax.annotate("Split Original Image", xy=(5,9.5), fontweight="bold", horizontalalignment="center")


def init():
	for p, loc in zip(patches, patches_start_locs):
		ax.add_patch(p)
		p.set_xy(loc)
	for p, loc in zip(c_squares, c_squares_locs):
		p.set_xy(loc)
	for p in new_squares:
		p.set_width(1.75)

	c_square_coll.set_paths(c_squares)
	c_square_coll.set_alpha(0.0)
	
	new_square_coll.set_paths(new_squares)
	new_square_coll.set_alpha(0.0)
	
	ax.add_collection(c_square_coll)
	ax.add_collection(new_square_coll)
		
	
	return patches + [c_square_coll, new_square_coll, text]
	
	
# animation schedule
# frames 0-2:
# reduce alpha on big outline
# frames 3-6
# move bottom and top rectangles away
# frames 7-9
# bring classify sqaures into frame
# frames 10
# begin new color in 
# frames 11-20
# classify turning new color
# frames 21-23
# fade classify out
# frames 24-27
# move rectangles back


	
def animate(frame_idx):
	if frame_idx<=2:	
		return patches + [c_square_coll, new_square_coll]

	if frame_idx<=6:
		rect_1.set_y(rect_1.get_y()-0.25)
		rect_3.set_y(rect_3.get_y()+0.25)
	
	elif frame_idx<=9:
		c_square_coll.set_zorder(99)
		c_square_coll.set_alpha(c_square_coll.get_alpha() + 0.33)

	elif frame_idx==10:
		for p in patches:
			p.zorder = 97
		text.set_text("Classify Sub Images")
		new_square_coll.set_alpha(1.0)
		
		new_square_coll.set_zorder(98)
	
	elif frame_idx<=20:
		for p in c_squares:
			p.set_x(p.get_x()+0.5)
		for p in new_squares:
			p.set_width(min(p.get_width()+0.5, 6))
		c_square_coll.set_paths(c_squares)
		new_square_coll.set_paths(new_squares)
	
		
	elif frame_idx<=23:
		c_square_coll.set_alpha(c_square_coll.get_alpha() - 0.33)
		
	elif frame_idx<=27:
		text.set_text("Merge Sub Images")
		for p in [rect_1, rect_2, rect_3]:
			p.set_alpha(0)
			
		new_square1.set_y(new_square1.get_y()-0.25)
		new_square3.set_y(new_square3.get_y()+0.25)
		
		new_square_coll.set_paths(new_squares)
		
	
	return patches + [c_square_coll, new_square_coll, text]


anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=30, 
                               interval=125,
                               blit=True)

anim.save('classify.gif', writer='imagemagick')

#plt.show()

