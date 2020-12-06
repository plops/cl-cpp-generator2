import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.ion()
import numpy as np
from b.cgal_mesher import *
_code_git_version="f8c13ae8a23f28c5f61e124670b43d631fe817fb"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="16:06:01 of Sunday, 2020-12-06 (GMT+1)"
cdt=ConstrainedDelaunayTriangulation()
coords_x=[]
coords_y=[]
va=cdt.insert(Point(100, 269))
vb=cdt.insert(Point(246, 269))
vc=cdt.insert(Point(246, 223))
vd=cdt.insert(Point(303, 223))
ve=cdt.insert(Point(303, 298))
vf=cdt.insert(Point(246, 298))
vg=cdt.insert(Point(246, 338))
vh=cdt.insert(Point(355, 338))
vi=cdt.insert(Point(355, 519))
vj=cdt.insert(Point(551, 519))
vk=cdt.insert(Point(551, 445))
vl=cdt.insert(Point(463, 445))
vm=cdt.insert(Point(463, 377))
vn=cdt.insert(Point(708, 377))
vo=cdt.insert(Point(708, 229))
vp=cdt.insert(Point(435, 229))
vq=cdt.insert(Point(435, 100))
vr=cdt.insert(Point(100, 100))
vs=cdt.insert(Point(349, 236))
vt=cdt.insert(Point(370, 236))
vu=cdt.insert(Point(370, 192))
vv=cdt.insert(Point(403, 192))
vw=cdt.insert(Point(403, 158))
vx=cdt.insert(Point(349, 158))
vy=cdt.insert(Point(501, 336))
vz=cdt.insert(Point(533, 336))
v1=cdt.insert(Point(519, 307))
v2=cdt.insert(Point(484, 307))
beg=va
end=vb
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vb
end=vc
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vc
end=vd
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vd
end=ve
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=ve
end=vf
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vf
end=vg
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vg
end=vh
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vh
end=vi
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vi
end=vj
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vj
end=vk
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vk
end=vl
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vl
end=vm
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vm
end=vn
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vn
end=vo
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vo
end=vp
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vp
end=vq
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vq
end=vr
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vr
end=va
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vs
end=vt
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vt
end=vu
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vu
end=vv
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vv
end=vw
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vw
end=vx
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vx
end=vs
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vy
end=vz
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=vz
end=v1
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=v1
end=v2
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
beg=v2
end=vy
cdt.insert_constraint(beg, end)
coords_x.append(beg.point.x)
coords_y.append(beg.point.y)
coords_x.append(end.point.x)
coords_y.append(end.point.y)
print("number of vertices: {}".format(cdt.number_of_vertices()))
mesher=Mesher(cdt)
seeds=(Point(505, 325),Point(379, 172),)
mesher.seeds_from(seeds)
make_conforming_delaunay(cdt)
print("number of vertices: {}".format(cdt.number_of_vertices()))
make_conforming_gabriel(cdt)
print("number of vertices: {}".format(cdt.number_of_vertices()))
mesher.criteria.aspect_bound=(0.1250    )
mesher.criteria.size_bound=(20.    )
mesher.refine_mesh()
print("number of vertices: {}".format(cdt.number_of_vertices()))
lloyd_optimize(cdt, max_iteration_number=10)
print("number of vertices: {}".format(cdt.number_of_vertices()))
print_faces_iterator_value_type()
point_to_index_map={vertex.point: idx for idx, vertex in enumerate(cdt.finite_vertices())}
triangles_idx=list(tuple(point_to_index_map[face.vertex_handle(i).point] for i in range(3)) for face in cdt.finite_faces())
triangles=np.array(list(tuple((face.vertex_handle(i).point.x,face.vertex_handle(i).point.y,) for i in range(3)) for face in cdt.finite_faces()))
plt.figure(0, (16,9,))
g=plt.gca()
for i in range(triangles.shape[0]):
    tri=plt.Polygon(triangles[i,:,:], facecolor=None, edgecolor="k", aa=True, color=None, fill=False, linewidth=(0.20    ))
    g.add_patch(tri)
plt.plot(coords_x, coords_y)
plt.scatter((505,379,), (325,172,), c="r", label="seed")
plt.legend()
plt.show()
plt.grid()
plt.xlim((0,800,))
plt.ylim((0,600,))