from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBList, PDBIO, Select
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import svm
from tqdm import tqdm
from math import sqrt, radians, sin, cos
from mpl_toolkits.mplot3d import Axes3D
import os


class SelectAtoms:
    @staticmethod
    def analyze_csv_files(csv_dir):
        output_img_dir = os.path.join(os.path.dirname(csv_dir))
        os.makedirs(output_img_dir, exist_ok=True)
        for filename in os.listdir(csv_dir):
            if not filename.endswith('.csv'):
                continue
            struct_id = filename[:-4]
            csv_path = os.path.join(csv_dir, filename)
            Atoms_coord_df = pd.read_csv(csv_path)
            if len(set(Atoms_coord_df['Chain_num'].values)) < 2:
                log_path = os.path.join(csv_dir, 'Diff_struct_10.csv')
                with open(log_path, 'a', encoding='utf-8') as w_file:
                    w_file.write(f"{struct_id}\n")
                continue
            svm_clf = svm.SVC(kernel='linear')
            svm_clf.fit(Atoms_coord_df[['x', 'y', 'z']].values, Atoms_coord_df['Chain_num'].astype('int'))
            plane_bind = list(svm_clf.coef_[0])
            plane_bind.append(svm_clf.intercept_[0])
            distant = 10
            abs_d = distant * (np.sqrt(plane_bind[0] ** 2 + plane_bind[1] ** 2 + plane_bind[2] ** 2))
            coeff_up = -abs_d + plane_bind[3]
            coeff_down = abs_d + plane_bind[3]
            plane_10_1 = [plane_bind[0], plane_bind[1], plane_bind[2], coeff_down]
            plane_10_2 = [plane_bind[0], plane_bind[1], plane_bind[2], coeff_up]
            center_1 = [0, 0, 0]
            x1, y1, z1, k1 = 0, 0, 0, 0
            center_2 = [0, 0, 0]
            x2, y2, z2, k2 = 0, 0, 0, 0
            Atoms_3A_1 = pd.DataFrame(columns=['Num', 'x', 'y', 'z', 'd'])
            Atoms_3A_2 = pd.DataFrame(columns=['Num', 'x', 'y', 'z', 'd'])
            Atoms_coord = pd.DataFrame(columns=['Num', 'x', 'y', 'z'])
            for j in range(Atoms_coord_df.shape[0]):
                d_down = plane_10_1[0] * Atoms_coord_df['x'][j] + plane_10_1[1] * Atoms_coord_df['y'][j] + plane_10_1[2] * Atoms_coord_df['z'][j] + plane_10_1[3]
                d_up = plane_10_2[0] * Atoms_coord_df['x'][j] + plane_10_2[1] * Atoms_coord_df['y'][j] + plane_10_2[2] * Atoms_coord_df['z'][j] + plane_10_2[3]
                if abs(d_down) < 1:
                    Atoms_3A_1.loc[len(Atoms_3A_1.index)] = [j, Atoms_coord_df['x'][j], Atoms_coord_df['y'][j], Atoms_coord_df['z'][j], d_down]
                if abs(d_up) < 1:
                    Atoms_3A_2.loc[len(Atoms_3A_2.index)] = [j, Atoms_coord_df['x'][j], Atoms_coord_df['y'][j], Atoms_coord_df['z'][j], d_up]
                if d_down >= 0 and d_up <= 0:
                    Atoms_coord.loc[len(Atoms_coord.index)] = [j, Atoms_coord_df['x'][j], Atoms_coord_df['y'][j], Atoms_coord_df['z'][j]]
                    if Atoms_coord_df['Chain_num'][j] == 0:
                        x1 += Atoms_coord_df['x'][j]
                        y1 += Atoms_coord_df['y'][j]
                        z1 += Atoms_coord_df['z'][j]
                        k1 += 1
                    elif Atoms_coord_df['Chain_num'][j] == 1:
                        x2 += Atoms_coord_df['x'][j]
                        y2 += Atoms_coord_df['y'][j]
                        z2 += Atoms_coord_df['z'][j]
                        k2 += 1
            if k1 == 0 or k2 == 0:
                with open(log_path, 'a') as w_file:
                    w_file.write(f"{struct_id}\n")
                continue
            center_1 = [round(x1 / k1), round(y1 / k1), round(z1 / k1)]
            center_2 = [round(x2 / k2), round(y2 / k2), round(z2 / k2)]
            Atoms_3A = Atoms_3A_2 if Atoms_3A_2.shape[0] >= Atoms_3A_1.shape[0] else Atoms_3A_1
            if Atoms_3A.shape[0] < 2:
                with open(log_path, 'a') as w_file:
                    w_file.write(f"{struct_id}\n")
                continue
            dots_max_dist = [0, 1, sqrt(SelectAtoms.dist(Atoms_3A.iloc[0], Atoms_3A.iloc[1]))]
            dot_is = SelectAtoms.isect_line_plane_v3_4d(center_1, center_2, plane_bind)
            if dot_is is None:
                continue
            x0, y0, z0 = dot_is[0], dot_is[1], dot_is[2]
            x1, y1, z1 = center_1[0], center_1[1], center_1[2]
            x2, y2, z2 = center_2[0], center_2[1], center_2[2]
            n_v = [x1 - x0, y1 - y0, z1 - z0]
            vec_3 = [[Atoms_3A['x'][dots_max_dist[0]] - Atoms_3A['x'][dots_max_dist[1]],
                      Atoms_3A['y'][dots_max_dist[0]] - Atoms_3A['y'][dots_max_dist[1]],
                      Atoms_3A['z'][dots_max_dist[0]] - Atoms_3A['z'][dots_max_dist[1]]]]
            u = np.array(vec_3)
            n = np.array([plane_bind[0], plane_bind[1], plane_bind[2]])
            n_norm = np.sqrt(sum(n ** 2))
            proj_of_u_on_n = (np.dot(u, n) / n_norm ** 2) * n
            proj_vec = u - proj_of_u_on_n
            proj_vec = np.reshape(proj_vec, -1)
            norm = plane_bind[:3]
            a = sqrt(1 / sum([coor ** 2 for coor in norm]))
            norm = [a * coor for coor in norm]
            norm = np.array(norm)
            a = sqrt(1 / sum([coor ** 2 for coor in proj_vec]))
            proj_vec = a * proj_vec
            last_vector = np.cross(norm, proj_vec)
            Atoms_coord = Atoms_coord_df.copy()
            Atoms_coord['x'] = Atoms_coord['x'] - dot_is[0]
            Atoms_coord['y'] = Atoms_coord['y'] - dot_is[1]
            Atoms_coord['z'] = Atoms_coord['z'] - dot_is[2]
            from_bazis = np.concatenate((np.expand_dims(proj_vec, axis=1),
                                         np.expand_dims(last_vector, axis=1),
                                         np.expand_dims(norm, axis=1)), axis=1)
            to_bazis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            s = np.matmul(np.linalg.inv(from_bazis), to_bazis)
            x_new = []
            y_new = []
            z_new = []
            for i, item in Atoms_coord.iterrows():
                vec = np.array([[item['x']], [item['y']], [item['z']]])
                new_vec = np.matmul(s, vec)
                new_xy = SelectAtoms.rotate_point_wrt_center((new_vec[0][0], new_vec[1][0]), 0)
                x_new.append(new_xy[0])
                y_new.append(new_xy[1])
                z_new.append(new_vec[2][0])
            Atoms_coord['x_new'] = x_new
            Atoms_coord['y_new'] = y_new
            Atoms_coord['z_new'] = z_new
            high_gr = 20
            length_gr = 40
            width_gr = 40
            high_v = [[0, 0, -high_gr], [0, 0, high_gr]]
            length_v = [[-length_gr, 0, 0], [length_gr, 0, 0]]
            width_v = [[0, -width_gr, 0], [0, width_gr, 0]]
            fig = plt.figure(figsize=(16, 10))
            ax = plt.axes(projection="3d")
            ax.plot3D([high_v[0][0], high_v[1][0]], [high_v[0][1], high_v[1][1]], [high_v[0][2], high_v[1][2]], 'red', linewidth=5)
            ax.plot3D([length_v[0][0], length_v[1][0]], [length_v[0][1], length_v[1][1]], [length_v[0][2], length_v[1][2]], 'green', linewidth=5)
            ax.plot3D([width_v[0][0], width_v[1][0]], [width_v[0][1], width_v[1][1]], [width_v[0][2], width_v[1][2]], 'blue', linewidth=5)
            ax.scatter3D(Atoms_coord['x_new'], Atoms_coord['y_new'], Atoms_coord['z_new'], c=Atoms_coord['Chain_num'])
            SelectAtoms.plot_linear_cube(ax, -length_gr, -width_gr, -high_gr, length_gr * 2, width_gr * 2, high_gr * 2, color='gray')
            ax.view_init(3, 40)
            plt.savefig(os.path.join(output_img_dir, f"{struct_id}.png"))
            plt.close(fig)
            Cell_atoms = pd.DataFrame(columns=['Chain_name', 'Chain_num', 'Residue', 'Type', 'Atom_id', 'x_new', 'y_new', 'z_new'])
            for id_atom in range(Atoms_coord.shape[0]):
                at_coord = Atoms_coord.iloc[id_atom]
                if abs(at_coord['x_new']) <= length_gr and abs(at_coord['y_new']) <= width_gr and abs(at_coord['z_new']) <= high_gr:
                    x_ind = at_coord['x_new'] + length_gr
                    y_ind = at_coord['y_new'] + width_gr
                    z_ind = at_coord['z_new'] + high_gr
                    Cell_atoms.loc[len(Cell_atoms.index)] = [at_coord['Chain_name'], at_coord['Chain_num'], at_coord['Residue'], at_coord['Type'], id_atom + 1, x_ind, y_ind, z_ind]
            if Cell_atoms.shape[0] < 10:
                new_log_path = os.path.join(csv_dir, 'Diff_struct_new_10.csv')
                with open(new_log_path, 'a', encoding='utf-8') as w_file:
                    w_file.write(f"{struct_id}\n")
            Cell_atoms_10 = pd.DataFrame(columns=['Index', 'Chain_name', 'Chain_num', 'Residue', 'Type', 'Atom_id', 'x_new', 'y_new', 'z_new'])
            atoms_chain1 = Cell_atoms[Cell_atoms['Chain_num'] == 0]
            atoms_chain2 = Cell_atoms[Cell_atoms['Chain_num'] == 1]
            for i in atoms_chain1.index.values:
                atom1 = atoms_chain1.loc[i]
                for j in atoms_chain2.index.values:
                    atom2 = atoms_chain2.loc[j]
                    b = [atom1['x_new'] - atom2['x_new'], atom1['y_new'] - atom2['y_new'], atom1['z_new'] - atom2['z_new']]
                    b_norm = np.linalg.norm(b)
                    if b_norm <= 10:
                        if i not in Cell_atoms_10['Index'].values:
                            Cell_atoms_10.loc[len(Cell_atoms_10.index)] = [i, atom1['Chain_name'], atom1['Chain_num'], atom1['Residue'], atom1['Type'], atom1['Atom_id'], atom1['x_new'], atom1['y_new'], atom1['z_new']]
                        if j not in Cell_atoms_10['Index'].values:
                            Cell_atoms_10.loc[len(Cell_atoms_10.index)] = [j, atom2['Chain_name'], atom2['Chain_num'], atom2['Residue'], atom2['Type'], atom2['Atom_id'], atom2['x_new'], atom2['y_new'], atom2['z_new']]
            final_csv_path = os.path.join(csv_dir, f"{struct_id}.csv")
            Cell_atoms_10.to_csv(final_csv_path, index=False)

    @staticmethod
    def isect_line_plane_v3_4d(p0, p1, plane, epsilon=1e-6):
        u = SelectAtoms.sub_v3v3(p1, p0)
        dot = SelectAtoms.dot_v3v3(plane, u)
        if abs(dot) > epsilon:
            p_co = SelectAtoms.mul_v3_fl(plane, -plane[3] / SelectAtoms.len_squared_v3(plane))
            w = SelectAtoms.sub_v3v3(p0, p_co)
            fac = -SelectAtoms.dot_v3v3(plane, w) / dot
            u = SelectAtoms.mul_v3_fl(u, fac)
            return SelectAtoms.add_v3v3(p0, u)
        return None

    @staticmethod
    def rotate_point_wrt_center(point_to_be_rotated, angle, center_point=(0, 0)):
        angle = radians(angle)
        xnew = cos(angle) * (point_to_be_rotated[0] - center_point[0]) - sin(angle) * (point_to_be_rotated[1] - center_point[1]) + center_point[0]
        ynew = sin(angle) * (point_to_be_rotated[0] - center_point[0]) + cos(angle) * (point_to_be_rotated[1] - center_point[1]) + center_point[1]
        return [xnew, ynew]

    @staticmethod
    def plot_linear_cube(ax, x, y, z, dx, dy, dz, color='red'):
        xx = [x, x, x + dx, x + dx, x]
        yy = [y, y + dy, y + dy, y, y]
        kwargs = {'alpha': 1, 'color': color}
        ax.plot3D(xx, yy, [z] * 5, **kwargs)
        ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
        ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
        ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
        ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)

    @staticmethod
    def add_v3v3(v0, v1):
        return (v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2])

    @staticmethod
    def sub_v3v3(v0, v1):
        return (v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2])

    @staticmethod
    def dot_v3v3(v0, v1):
        return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]

    @staticmethod
    def len_squared_v3(v0):
        return SelectAtoms.dot_v3v3(v0, v0)

    @staticmethod
    def mul_v3_fl(v0, f):
        return (v0[0] * f, v0[1] * f, v0[2] * f)

    @staticmethod
    def dist(p1, p2):
        x0 = p1[0] - p2[0]
        y0 = p1[1] - p2[1]
        z0 = p1[2] - p2[2]
        return x0 * x0 + y0 * y0 + z0 * z0

