# This is a remeshing algorithm that only uses blender-native operations.
# It Works in a similar fashion to Toporake feature of dyntopo.

bl_info = {
    "name": "Boundary Aligned Remesh",
    "author": "Jean Da Costa",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > W > ",
    "description": "Rebuilds mesh out of isotropic polygons.",
    "warning": "",
    "wiki_url": "",
    "category": "Remesh",
}

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# Main Remesher class, this stores all the needed data
class BoundaryAlignedRemesher:

    def __init__(self, obj):
        self.obj = obj
        mode = obj.mode
        self.edit_mode = mode == 'EDIT'

        # hack to update the mesh data
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode=mode)

        self.bm = bmesh.new()

        self.bm.from_mesh(obj.data)
        self.bm1 = None

        if self.edit_mode:
            self.bm1 = self.bm.copy()
            remove, remove1 = [], []

            for vert in self.bm.verts:
                if all(not f.select for f in vert.link_faces):
                    remove.append(vert)
            for vert in remove:
                self.bm.verts.remove(vert)

            for vert in self.bm1.verts:
                if all(v.select for v in vert.link_faces):
                    remove1.append(vert)
            for vert in remove1:
                self.bm1.verts.remove(vert)

            remove1 = [f for f in self.bm1.faces if f.select]
            for face in remove1:
                self.bm1.faces.remove(face)

        self.bvh = BVHTree.FromBMesh(self.bm)

        # Boundary_data is a list of directions and locations of boundaries.
        # This data will serve as guidance for the alignment
        self.boundary_data = []

        # Fill the data using boundary edges as source of directional data.
        for edge in self.bm.edges:
            if edge.is_boundary:
                vec = (edge.verts[0].co - edge.verts[1].co).normalized()
                center = (edge.verts[0].co + edge.verts[1].co) / 2

                self.boundary_data.append((center, vec))

        # Create a Kd Tree to easily locate the nearest boundary point
        self.boundary_kd_tree = KDTree(len(self.boundary_data))

        for index, (center, vec) in enumerate(self.boundary_data):
            self.boundary_kd_tree.insert(center, index)

        self.boundary_kd_tree.balance()

    def nearest_boundary_vector(self, location):
        """ Gets the nearest boundary direction """
        location, index, dist = self.boundary_kd_tree.find(location)
        location, vec = self.boundary_data[index]
        return vec

    def enforce_edge_length(self, edge_length=0.05, bias=0.333):
        """ Replicates dyntopo behavior """
        upper_length = edge_length + edge_length * bias
        lower_length = edge_length - edge_length * bias

        # Subdivide Long edges
        subdivide = []
        for edge in self.bm.edges:
            if edge.calc_length() > upper_length:
                subdivide.append(edge)

        bmesh.ops.subdivide_edges(self.bm, edges=subdivide, cuts=1)
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces)

        if self.edit_mode:
            subdivide = []
            for edge in self.bm1.edges:
                if edge.select and edge.calc_length() > upper_length:
                    subdivide.append(edge)

            bmesh.ops.subdivide_edges(self.bm1, edges=subdivide, cuts=1)

        # Remove verts with less than 5 edges, this helps inprove mesh quality
        dissolve_verts = []
        for vert in self.bm.verts:
            if len(vert.link_edges) < 5:
                if not vert.is_boundary:
                    dissolve_verts.append(vert)

        bmesh.ops.dissolve_verts(self.bm, verts=dissolve_verts)
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces)

        # Collapse short edges but ignore boundaries and never collapse two chained edges
        lock_verts = set(vert for vert in self.bm.verts if vert.is_boundary)
        collapse = []

        for edge in self.bm.edges:
            if edge.calc_length() < lower_length and not edge.is_boundary:
                verts = set(edge.verts)
                if verts & lock_verts:
                    continue
                collapse.append(edge)
                lock_verts |= verts

        bmesh.ops.collapse(self.bm, edges=collapse)
        bmesh.ops.beautify_fill(self.bm, faces=self.bm.faces, method="ANGLE")

    def align_verts(self, rule=(-1, -2, -3, -4)):
        # Align verts to the nearest boundary by averaging neigbor vert locations selected
        # by a specific rule,

        # Rules work by sorting edges by angle relative to the boundary.
        # Eg1. (0, 1) stands for averagiing the biggest angle and the 2nd biggest angle edges.
        # Eg2. (-1, -2, -3, -4), averages the four smallest angle edges
        for vert in self.bm.verts:
            if not vert.is_boundary:

                # min_edge = min(vert.link_edges, key=lambda e: e.calc_length())
                # other = min_edge.other_vert(vert)
                # vec = other.co - vert.co
                # vert.co -= vec * 0.1
                #

                vec = self.nearest_boundary_vector(vert.co)
                neighbor_locations = [edge.other_vert(vert).co for edge in vert.link_edges]
                best_locations = sorted(neighbor_locations,
                                        key = lambda n_loc: abs((n_loc - vert.co).normalized().dot(vec)))
                co = vert.co.copy()
                le = len(vert.link_edges)
                for i in   rule:
                    co += best_locations[i % le]
                co /= len(rule) + 1
                co -= vert.co
                co -= co.dot(vert.normal) * vert.normal
                vert.co += co

        self.reproject()

    def reproject(self):
        """ Recovers original shape """
        for vert in self.bm.verts:
            if vert.is_boundary:
                continue
            location, normal, index, dist = self.bvh.find_nearest(vert.co)
            if location:
                vert.co = location

    def remesh(self,edge_length=0.05, iterations=30, quads=True):
        """ Coordenates remeshing """

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='OBJECT')

        if quads:
            rule = (-1,-2, 0, 1)
        else:
            rule = (0, 1, 2, 3)

        for _ in range(iterations):
            self.enforce_edge_length(edge_length=edge_length)
            self.align_verts(rule=rule)
            self.reproject()

        if quads:
            bmesh.ops.join_triangles(self.bm, faces=self.bm.faces,
                                     angle_face_threshold=3.14,
                                     angle_shape_threshold=3.14)

        for vert in self.bm.verts:
            vert.select = True
        for face in self.bm.faces:
            face.select = True

        if self.bm1:
            self.bm1.to_mesh(self.obj.data)
            self.bm.from_mesh(self.obj.data)

            bmesh.ops.remove_doubles(self.bm, verts=[v for v in self.bm.verts if v.select], dist=0.00001)

        self.bm.to_mesh(self.obj.data)

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')

class Remesher(bpy.types.Operator):
    bl_idname = "remesh.boundary_aligned_remesh"
    bl_label = "Boundary Aligned Remesh"
    bl_options = {"REGISTER", "UNDO"}

    resolition: bpy.props.FloatProperty(
        name="Resolution",
        min=1,
        default = 30
    )

    iterations: bpy.props.IntProperty(
        name="Iterations",
        min=1,
        default=30
    )

    quads: bpy.props.BoolProperty(
        name="Quads",
        default=False
    )

    def execute(self, context):
        obj = bpy.context.active_object
        print(f"Remeshing {obj.name}")

        size = max(d / s for d, s in zip(obj.dimensions, obj.scale))
        edge_length = size / self.resolition

        last_mode = obj.mode
        remesher = BoundaryAlignedRemesher(obj)
        remesher.remesh(edge_length, self.iterations, self.quads)
        context.area.tag_redraw()
        return {"FINISHED"}

def draw(self, context):
    self.layout.operator("remesh.boundary_aligned_remesh", text="Boundary Aligned Remesh")

def register():
    bpy.utils.register_class(Remesher)
    bpy.types.VIEW3D_MT_object_context_menu.append(draw)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(draw)

def unregister():
    bpy.utils.unregister_class(Remesher)
    bpy.types.VIEW3D_MT_object_context_menu.remove(draw)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(draw)

if __name__ == "__main__":
    register()
