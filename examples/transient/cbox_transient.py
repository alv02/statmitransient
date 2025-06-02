# If you have compiled Mitsuba 3 yourself, you will need to specify the path
# to the compilation folder
# import sys
# sys.path.insert(0, '<mitsuba-path>/mitsuba3/build/python')
import mitsuba as mi

# To set a variant, you need to have set it in the mitsuba.conf file
# https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html
mi.set_variant("llvm_ad_rgb")

# Load XML file
# You can also use mi.load_dict and pass a Python dict object
# but it is probably much easier for your work to use XML files
import os

import mitransient as mitr

scene = mi.load_file(os.path.abspath("cornell-box/cbox_diffuse.xml"))
data_steady, data_transient = mi.render(scene, spp=32)


# Plot the computed steady image

print(data_transient.shape)
