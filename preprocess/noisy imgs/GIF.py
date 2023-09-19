'''
Create GIF given pngs
'''



from PIL import Image
import glob

# Create the frames
frames = []
imgs = glob.glob("*.png")
imgs.sort()
# print(imgs)
for i in imgs:
    print(i)
    new_frame = Image.open(i)
    frames.append(new_frame)
#frames = [frame.convert('P') for frame in frames]

# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=400, loop=0)

# import imageio
# images = []
# for filename in imgs:
#     images.append(imageio.v2.imread(filename))
# imageio.mimsave('movie.gif', images)