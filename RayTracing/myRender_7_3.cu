#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <time.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Magic.hpp"

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR: %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

__host__ __device__ float my_fmax(float x, float y) {
    return x > y ? x : y;
}

__host__ __device__ float my_fmin(float x, float y) {
    return x < y ? x : y;
}

__host__ __device__ float operator*(const float3 &lhs, const float3 &rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ float3 operator+(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__host__ __device__ float3 operator-(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__ float3 operator-(const float3 &lhs) {
    return make_float3(-1 * lhs.x, -1 * lhs.y, -1 * lhs.z);
}

__host__ __device__ float3 operator*(const float3 &lhs, const float &rhs) {
    return make_float3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

__host__ __device__ float norm(const float3 &lhs) {
    return sqrtf(lhs.x*lhs.x + lhs.y*lhs.y + lhs.z*lhs.z);
}

__host__ __device__ float3 normalize(const float3 &lhs) {
    return lhs * (1 / norm(lhs));
}

__host__ __device__ float3 cross(const float3 &v1, const float3 &v2) {
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct Light {
    __host__ __device__ Light(const float3 &p, const float i) : position(p), intensity(i) {}
    float3 position;
    float intensity;
};

struct Material {
    __host__ __device__ Material(const float r, const float4 &a, const float3 &color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    __host__ __device__ Material() : refractive_index(1), albedo(make_float4(1,0,0,0)), diffuse_color(), specular_exponent() {}
    float refractive_index;
    float4 albedo;
    float3 diffuse_color;
    float specular_exponent;
};

struct Ray {
    float3 orig;
    float3 dir;
    float3 N;
    float3 color;
    float3 point;
    Material material;
    int depth;
    bool hited;
};

struct Camera {
    int id;
    
    float3 eye; 
    float3 gaze;

    int width;
    int height;
    int fow;
    
    float aspectRatio;
    float scale;
    
    __host__ __device__ Camera(int n, float rc, float phic, float zc, float rn, float phin, float zn, int px_w, int px_h, int fw) {
        id = n;
        
        eye = make_float3(rc * cos(phic), zc, -rc * sin(phic));
        gaze = make_float3(rn * cos(phin), zn, -rn * sin(phin));
        
        width = px_w;
        height = px_h;
        fow = fw * M_PI / 180;
        scale = tan(fow * 0.5);
        aspectRatio = width / (float)height;
    }
    
    __host__ __device__ Matrix44<float> cameraToWorld() {
        float3 u, v, w, t;
        t = make_float3(0, 1, 0);
        w = normalize((eye - gaze));
        u = normalize((cross(t, w)));
        v = cross(w, u);
        
        float m[4][4] = {{u.x, u.y, u.z, 0},
                         {v.x, v.y, v.z, 0},
                         {w.x, w.y, w.z, 0},
                         {eye.x, eye.y, eye.z, 1}};
        return Matrix44<float>(m);
    }
    
};

__host__ __device__ float3 reflect(const float3 &I, const float3 &N) {
    return I - N*2.f*(I*N);
}

__host__ __device__ float3 refract(const float3 &I, const float3 &N, const float eta_t, const float eta_i=1.f) { // Snell's law
    float cosi = - my_fmax(-1.f, my_fmin(1.f, I*N));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? make_float3(1, 0, 0) : I*eta + N*(eta*cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

struct Sphere {
    float3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere(const float3 &c, const float r, const Material &m) : center(c), radius(r), material(m) {}

    __host__ __device__ bool ray_intersect(const float3 &orig, const float3 &dir, float &t0) const {
        float3 L = center - orig;//                  вектор из точки обзора к центру сферы
        float tca = L * dir;//                      расстояние от точки обзора до проекции центра на луч
        float d2 = L * L - tca * tca;//             квадрат расстояния от центра сферы до проекции центра сферы на луч
        if (d2 > radius * radius) return false;//   луч не пересекает сферу
        float thc = sqrtf(radius * radius - d2);//  расстояние от точки пересечения сферы до проекции центра на луч
        t0 = tca - thc;//                           расстояние от точки обзора до ближайшей точки пересечения со сферой
        float t1 = tca + thc;//                     расстояние до дальнейшей точки пересечения
        if (t0 < 0) t0 = t1;//                      ближайшее пересечение за точкой обзора
        if (t0 < 0) return false;//                 второе пересечение тоже там => сферу не видно, она за нами полностью
        return true;//                              сфера либо перед нами, либо мы внутри неё, т.е. её видно
    }
};

__host__ __device__ bool ray_intersect_face(const float3& orig,
                        const float3& dir,
                        const float3& v0,
                        const float3& v1,
                        const float3& v2,
                        float &t0) {    
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    // Вычисление вектора нормали к плоскости (не совсем)
    float3 pvec = cross(dir, e2);
    // float det = dot(e1, pvec);
    float det = e1 * pvec;

    // Луч параллелен плоскости
    if (det < 1e-8 && det > -1e-8) {
        t0 = 10000.;
        return false;
    }

    float inv_det = 1 / det;
    float3 tvec = orig - v0;
    float u = (tvec * pvec) * inv_det;
    if (u < 0 || u > 1) {
        t0 = 10000.;
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = (dir * qvec) * inv_det;
    if (v < 0 || u + v > 1) {
        t0 = 10000;
        return false;
    }
    t0 = (e2 * qvec) * inv_det;
    return true;
}

struct Tetrahedron {//                  правильная черыхехгранная пирамида
    float3 center;
    float side;
    Material material;

    float3 S = make_float3(0, 0, 0);
    float3 A = make_float3(0, 0, 0);
    float3 B = make_float3(0, 0, 0);
    float3 C = make_float3(0, 0, 0);

    Tetrahedron(const float3 &c, const float &s, const Material &m) : center(c), side(s), material(m) {
        A.z = center.z;
        A.y = center.y;
        A.x = center.x - side / sqrt(3);
        
        S.z = center.z;
        S.x = center.x;
        S.y = center.y + sqrtf((float)2 / 3) * side;
        
        B.y = center.y;
        C.y = center.y;
        
        B.x = center.x + side / (2 * sqrt(3));
        C.x = center.x + side / (2 * sqrt(3));
        B.z = center.z - side / 2;
        C.z = center.z + side / 2;
    }
    
    Material material1[4] = {material, material, material, material};
        
    __host__ __device__ bool ray_intersect(const float3& orig,
                       const float3& dir,
                       float &t0,
                       float3 &N,
                       int &ind) const {
        float ts[4];
        
        t0 = 10000;
        bool tmp1 = !ray_intersect_face(orig, dir, S, A, B, ts[0]);
        bool tmp2 = !ray_intersect_face(orig, dir, S, B, C, ts[1]);
        bool tmp3 = !ray_intersect_face(orig, dir, S, A, C, ts[2]);
        bool tmp4 = !ray_intersect_face(orig, dir, A, B, C, ts[3]);
        
        if (tmp1 && tmp2 && tmp3 && tmp4) return false;

        t0 = ts[0];
        ind = 0;

        for (int i = 0; i < 4; i++) {
            if (ts[i] < t0) {
                t0 = ts[i];
                ind = i;
            }
        }

        if (ind == 0) {
            N = cross(normalize((B - S)), normalize((A - S)));
        } else if (ind == 1) {
            N = cross(normalize((B - S)), normalize((C - S))) * (-1);
        } else if (ind == 2) {
            N = cross(normalize((A - S)), normalize((C - S)));
        } else if (ind == 3) {
            N = cross(normalize((B - A)), normalize((C - A)));
        }
        
        N = normalize(N);

        return true;
    }
};

struct Cube {//                         куб
    float3 center;
    float side;
    Material material;

    float3 A = make_float3(0, 0, 0);//bot
    float3 B = make_float3(0, 0, 0);
    float3 C = make_float3(0, 0, 0);
    float3 D = make_float3(0, 0, 0);    
    float3 A1 = make_float3(0, 0, 0);//top
    float3 B1 = make_float3(0, 0, 0);
    float3 C1 = make_float3(0, 0, 0);
    float3 D1 = make_float3(0, 0, 0);

    Cube(const float3 &c, const float &s, const Material &m) : center(c), side(s), material(m) {
        A.y = center.y;
        B.y = center.y;
        C.y = center.y;
        D.y = center.y;
        
        A1.y = center.y + side;
        B1.y = center.y + side;
        C1.y = center.y + side;
        D1.y = center.y + side;
        
        float R = sqrt(2) * side / 2;
        A.x = center.x - R;
        B.x = center.x;
        C.x = center.x + R;
        D.x = center.x;
        
        A1.x = center.x - R;
        B1.x = center.x;
        C1.x = center.x + R;
        D1.x = center.x;
        
        A.z = center.z;
        B.z = center.z - R;
        C.z = center.z;
        D.z = center.z + R;
        
        A1.z = center.z;
        B1.z = center.z - R;
        C1.z = center.z;
        D1.z = center.z + R;
    }
            
    __host__ __device__ bool ray_intersect(const float3& orig,
                       const float3& dir,
                       float &t0,
                       float3 &N,
                       int &ind) const {
        float ts[12];
        t0 = 10000;

        bool tmp11 = !ray_intersect_face(orig, dir, A1, A, D, ts[0]);//left
        bool tmp12 = !ray_intersect_face(orig, dir, A1, D1, D, ts[1]);
        
        bool tmp21 = !ray_intersect_face(orig, dir, A1, A, B, ts[2]);//into
        bool tmp22 = !ray_intersect_face(orig, dir, A1, B1, B, ts[3]);
        
        bool tmp31 = !ray_intersect_face(orig, dir, B1, B, C, ts[4]);//right
        bool tmp32 = !ray_intersect_face(orig, dir, B1, C1, C, ts[5]);
        
        bool tmp41 = !ray_intersect_face(orig, dir, D1, C1, C, ts[6]);//outer
        bool tmp42 = !ray_intersect_face(orig, dir, D1, D, C, ts[7]);
        
        bool tmp51 = !ray_intersect_face(orig, dir, A1, B1, C1, ts[8]);//up
        bool tmp52 = !ray_intersect_face(orig, dir, A1, D1, C1, ts[9]);
        
        bool tmp61 = !ray_intersect_face(orig, dir, A, B, C, ts[10]);//down
        bool tmp62 = !ray_intersect_face(orig, dir, A, D, C, ts[11]);
        
        if (tmp11 && tmp12 &&
            tmp21 && tmp22 &&
            tmp31 && tmp32 &&
            tmp41 && tmp42 &&
            tmp51 && tmp52 &&
            tmp61 && tmp62) return false;

        t0 = ts[0];
        ind = 0;
        for (int i = 0; i < 12; i++) {
            if (ts[i] < t0) {
                t0 = ts[i];
                ind = i;
            }
        }
        
        if (ind == 0 || ind == 1) {
            N = cross(normalize((A1 - D1)), normalize((D - D1)));
        } else if (ind == 2 || ind == 3) {
            N = cross(normalize((A1 - A)), normalize((B - A)));
        } else if (ind == 4 || ind == 5) {
            N = cross(normalize((B1 - B)), normalize((C - B)));
        } else if (ind == 6 || ind == 7) {
            N = cross(normalize((D1 - C1)), normalize((C - C1)));
        } else if (ind == 8 || ind == 9) {
            N = cross(normalize((A1 - B1)), normalize((C1 - B1)));
        } else if (ind == 10 || ind == 11) {
            N = cross(normalize((A - B)), normalize((C - B))) * (-1);
        }
        
        N = normalize(N);

        return true;
    }
};

struct Icosahedron {//                  правильный десятигранник (икосаэдр)
    float3 center;
    float side;
    Material material;
    
    float k = M_PI / 180;
    
    float R=7.5; // радиус сферы

    // начальные значения
    float a=4*R/sqrt(10+2*sqrt(5)); // сторона икосаэдра
    float alpha=acos((1-a*a/2/R/R)); // первый угол поворота по тэта
    
    float3 VerMas[12];
    float3 IndMas[20];

    Icosahedron(const float3 &c, const float &s, const Material &m) : center(c), R(s), material(m) {
        a=4*R/sqrt(10+2*sqrt(5));
        alpha=acos((1-a*a/2/R/R));
        
        VerMas[0].x=0;   // x
        VerMas[0].y=0;   // y
        VerMas[0].z=R;   // z

        VerMas[1].x=R*sin(alpha)*sin(0);
        VerMas[1].y=R*sin(alpha)*cos(0);
        VerMas[1].z=R*cos(alpha);

        VerMas[2].x=R*sin(alpha)*sin(72*k);
        VerMas[2].y=R*sin(alpha)*cos(72*k);
        VerMas[2].z=R*cos(alpha);

        VerMas[3].x=R*sin(alpha)*sin(2*72*k);
        VerMas[3].y=R*sin(alpha)*cos(2*72*k);
        VerMas[3].z=R*cos(alpha);

        VerMas[4].x=R*sin(alpha)*sin(3*72*k);
        VerMas[4].y=R*sin(alpha)*cos(3*72*k);
        VerMas[4].z=R*cos(alpha);

        VerMas[5].x=R*sin(alpha)*sin(4*72*k);
        VerMas[5].y=R*sin(alpha)*cos(4*72*k);
        VerMas[5].z=R*cos(alpha);

        VerMas[6].x=R*sin(M_PI-alpha)*sin(-36*k);
        VerMas[6].y=R*sin(M_PI-alpha)*cos(-36*k);
        VerMas[6].z=R*cos(M_PI-alpha);

        VerMas[7].x=R*sin(M_PI-alpha)*sin(36*k);
        VerMas[7].y=R*sin(M_PI-alpha)*cos(36*k);
        VerMas[7].z=R*cos(M_PI-alpha);

        VerMas[8].x=R*sin(M_PI-alpha)*sin((36+72)*k);
        VerMas[8].y=R*sin(M_PI-alpha)*cos((36+72)*k);
        VerMas[8].z=R*cos(M_PI-alpha);

        VerMas[9].x=R*sin(M_PI-alpha)*sin((36+2*72)*k);
        VerMas[9].y=R*sin(M_PI-alpha)*cos((36+2*72)*k);
        VerMas[9].z=R*cos(M_PI-alpha);

        VerMas[10].x=R*sin(M_PI-alpha)*sin((36+3*72)*k);
        VerMas[10].y=R*sin(M_PI-alpha)*cos((36+3*72)*k);
        VerMas[10].z=R*cos(M_PI-alpha);

        VerMas[11].x=0;
        VerMas[11].y=0;
        VerMas[11].z=-R;
        
        // задаем индексы каждого из треугольников
        IndMas[0].x=0; // индекс первой вершины
        IndMas[0].y=2; // индекс второй вершины
        IndMas[0].z=1; // индекс третьей вершины

        IndMas[1].x=0;
        IndMas[1].y=3;
        IndMas[1].z=2;

        IndMas[2].x=0;
        IndMas[2].y=4;
        IndMas[2].z=3;

        IndMas[3].x=0;
        IndMas[3].y=5;
        IndMas[3].z=4;

        IndMas[4].x=0;
        IndMas[4].y=1;
        IndMas[4].z=5;

        IndMas[5].x=6;
        IndMas[5].y=1;
        IndMas[5].z=7;

        IndMas[6].x=7;
        IndMas[6].y=1;
        IndMas[6].z=2;

        IndMas[7].x=7;
        IndMas[7].y=2;
        IndMas[7].z=8;

        IndMas[8].x=8;
        IndMas[8].y=2;
        IndMas[8].z=3;

        IndMas[9].x=8;
        IndMas[9].y=3;
        IndMas[9].z=9;

        IndMas[10].x=9;
        IndMas[10].y=3;
        IndMas[10].z=4;

        IndMas[11].x=9;
        IndMas[11].y=4;
        IndMas[11].z=10;

        IndMas[12].x=10;
        IndMas[12].y=4;
        IndMas[12].z=5;

        IndMas[13].x=10;
        IndMas[13].y=5;
        IndMas[13].z=6;

        IndMas[14].x=6;
        IndMas[14].y=5;
        IndMas[14].z=1;

        IndMas[15].x=7;
        IndMas[15].y=11;
        IndMas[15].z=6;

        IndMas[16].x=8;
        IndMas[16].y=11;
        IndMas[16].z=7;

        IndMas[17].x=9;
        IndMas[17].y=11;
        IndMas[17].z=8;

        IndMas[18].x=10;
        IndMas[18].y=11;
        IndMas[18].z=9;

        IndMas[19].x=6;
        IndMas[19].y=11;
        IndMas[19].z=10;
        
        for (int i = 0; i < 12; i++) {
            VerMas[i] = VerMas[i] + center;
        }
    }
    
    __host__ __device__ bool ray_intersect(const float3& orig,
                       const float3& dir,
                       float &t0,
                       float3 &N,
                       int &ind) const {
        float ts[20];
        t0 = 10000;
        
        bool tmps[20];
        int cnt = 0;
        for (int i = 0; i < 20; i++) {
            tmps[i] = !ray_intersect_face(orig, dir, VerMas[(int)IndMas[i].x], VerMas[(int)IndMas[i].y], VerMas[(int)IndMas[i].z], ts[i]);
            if (tmps[i] == true) cnt++;
        }
        if (cnt == 20) return false;

        t0 = ts[0];
        ind = 0;
        for (int i = 0; i < 20; i++) {
            if (ts[i] < t0) {
                t0 = ts[i];
                ind = i;
            }
        }
        
        N = cross(normalize((VerMas[(int)IndMas[ind].y] - VerMas[(int)IndMas[ind].x])), normalize((VerMas[(int)IndMas[ind].z] - VerMas[(int)IndMas[ind].x])));
        N = normalize(N);
        
        return true;
    }
};

__host__ __device__ bool scene_intersect1(const float3 &orig, const float3 &dir, const Sphere* spheres, int ssize, const Tetrahedron* tetrahedrones, int tsize,
                            const Cube* cubes, int csize, const Icosahedron* icosahedrones, int isize,
                            float3 &hit, float3 &N, Material &material) {
    float spheres_dist = 10000.;
    float dist_i;
    int ind;
    float3 N_tmp = N;
    for (size_t i=0; i < my_fmax(my_fmax(my_fmax(tsize, ssize), csize), isize); i++) {
        if (i < tsize && tetrahedrones[i].ray_intersect(orig, dir, dist_i, N_tmp, ind) && dist_i < spheres_dist) {

            if (dist_i > 0) {
                spheres_dist = dist_i;
                hit = orig + dir*dist_i;
                N = N_tmp;
                material = tetrahedrones[i].material1[ind];
            }
        }
        if (i < ssize && spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir*dist_i;
            N = normalize((hit - spheres[i].center));
            material = spheres[i].material;
        }
        if (i < csize && cubes[i].ray_intersect(orig, dir, dist_i, N_tmp, ind) && dist_i < spheres_dist) {
            if (dist_i > 0) {
                spheres_dist = dist_i;
                hit = orig + dir*dist_i;
                N = N_tmp;
                material = cubes[i].material;
            }
        }
        if (i < isize && icosahedrones[i].ray_intersect(orig, dir, dist_i, N_tmp, ind) && dist_i < spheres_dist) {
            if (dist_i > 0) {
                spheres_dist = dist_i;
                hit = orig + dir*dist_i;
                N = N_tmp;
                material = icosahedrones[i].material;
            }
        }
    }
    
    float checkerboard_dist = 10000.;
    if (fabs(dir.y)>1e-3)  {
        float d = -(orig.y+4)/dir.y; // the checkerboard plane has equation y = -4
        float3 pt = orig + dir*d;
        if (d>0 && fabs(pt.x)<10 && pt.z<10 && pt.z>-10 && d<spheres_dist) {
            checkerboard_dist = d;
            hit = pt;
            N = make_float3(0,1,0);
            material.diffuse_color = (int(.5*hit.x+1000) + int(.5*hit.z + 1000)) & 1 ? make_float3(.0, .1, .3) : make_float3(.25, .25, .3);
        }
    }
    
    return my_fmin(spheres_dist, checkerboard_dist)<1000;
}

__host__ __device__ void cast_ray1(float3 back_color, Ray &ray, const Sphere* spheres, int ssize, const Tetrahedron* tetrahedrones, int tsize,
                            const Cube* cubes, int csize, const Icosahedron* icosahedrones, int isize,
                            const Light* lights, int lsize) {
    float3 point, N;
    Material material;
    float3 orig = ray.orig;
    float3 dir = ray.dir;
    int depth = ray.depth;

    if (depth >= 8 || !scene_intersect1(orig, dir, spheres, ssize, tetrahedrones, tsize,
                                       cubes, csize, icosahedrones, isize,
                                       point, N, material)) {
        // ray.color = make_float3(0., 0., 0.); // background color
        ray.color = back_color; // background color
        // ray.color = make_float3(0.2, 0.7, 0.8); // background color
        ray.point = point;
        ray.N = N;
        ray.material = material;
        ray.hited = false;
        return;
    }

    ray.point = point;
    ray.N = N;
    ray.material = material;
    ray.hited = true;

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i=0; i<lsize; i++) {
        float3 light_dir      = normalize((lights[i].position - point));        
        float light_distance = norm((lights[i].position - point));
        
        float3 shadow_orig = light_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // checking if the point lies in the shadow of the lights[i]
        float3 shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect1(shadow_orig, light_dir, spheres, ssize, tetrahedrones, tsize, cubes, csize, icosahedrones, isize, shadow_pt, shadow_N, tmpmaterial) && norm((shadow_pt-shadow_orig)) < light_distance) {
           continue;
        }
        
        diffuse_light_intensity  += lights[i].intensity * my_fmax(0.f, light_dir * N);
        specular_light_intensity += powf(my_fmax(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent)*lights[i].intensity;
    }

    ray.color = material.diffuse_color * diffuse_light_intensity * material.albedo.x + make_float3(1., 1., 1.)*specular_light_intensity * material.albedo.y /*+ reflect_color*material.albedo.z + refract_color*material.albedo.w*/;
}

__global__ void kernel1(float3 back_color, Camera cam, int level, Ray* rays1, Ray* rays2, int width, int height, float3* framebuffer, const Sphere* spheres, int ssize, const Tetrahedron* tetrahedrones, int tsize,
                        const Cube* cubes, int csize, const Icosahedron* icosahedrones, int isize,
                        const Light* lights, int lsize) {
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    
    if (level == 1) {
        Matrix44<float> cameraToWorld = cam.cameraToWorld();
        float3 orig;
        float3 dir;

        cameraToWorld.multVecMatrix(make_float3(0, 0, 0), orig);
        
        const int fov = M_PI / 3.;
        for (int j = y; j < height; j += offset_y) {
            for (int i = x; i < width; i += offset_x) {
                float dir_x =  (2*(i + 0.5)/(float)width  - 1)*tan(fov/2.)*width/(float)height;
                float dir_y = -(2*(j + 0.5)/(float)height - 1)*tan(fov/2.);
                float dir_z = -1;
                
                cameraToWorld.multDirMatrix(make_float3(dir_x, dir_y, dir_z), dir);
                
                rays1[(i+j*width) * level].depth = 0;
                rays1[(i+j*width) * level].orig = orig;
                rays1[(i+j*width) * level].dir = normalize(dir);
                cast_ray1(back_color, rays1[(i+j*width) * level], spheres, ssize, tetrahedrones, tsize, cubes, csize, icosahedrones, isize, lights, lsize);
            }
        }
        __syncthreads();
    }
    
    for (int j = y; j < height; j += offset_y) {
        for (int i = x; i < width; i += offset_x) {
            for (int q = 0; q < level; q++) {
                rays2[2 * ((i+j*width) * level + q)].depth = level;
                rays2[2 * ((i+j*width) * level + q) + 1].depth = level;
                
                rays2[2 * ((i+j*width) * level + q)].dir = normalize(reflect(rays1[((i+j*width) * level + q)].dir, rays1[((i+j*width) * level + q)].N));
                rays2[2 * ((i+j*width) * level + q) + 1].dir = normalize(refract(rays1[((i+j*width) * level + q)].dir, rays1[((i+j*width) * level + q)].N, rays1[((i+j*width) * level + q)].material.refractive_index));
                
                rays2[2 * ((i+j*width) * level + q)].orig = rays2[2 * ((i+j*width) * level + q)].dir*rays1[((i+j*width) * level + q)].N < 0 ? rays1[((i+j*width) * level + q)].point - rays1[((i+j*width) * level + q)].N*1e-3 : rays1[((i+j*width) * level + q)].point + rays1[((i+j*width) * level + q)].N*1e-3;
                rays2[2 * ((i+j*width) * level + q) + 1].orig = rays2[2 * ((i+j*width) * level + q) + 1].dir*rays1[((i+j*width) * level + q)].N < 0 ? rays1[((i+j*width) * level + q)].point - rays1[((i+j*width) * level + q)].N*1e-3 : rays1[((i+j*width) * level + q)].point + rays1[((i+j*width) * level + q)].N*1e-3;
                
                cast_ray1(back_color, rays2[2 * ((i+j*width) * level + q)], spheres, ssize, tetrahedrones, tsize, cubes, csize, icosahedrones, isize, lights, lsize);
                __syncthreads();
                cast_ray1(back_color, rays2[2 * ((i+j*width) * level + q) + 1], spheres, ssize, tetrahedrones, tsize, cubes, csize, icosahedrones, isize, lights, lsize);
                __syncthreads();

                if (!rays1[((i+j*width) * level + q)].hited) {
                    rays2[2 * ((i+j*width) * level + q)].depth = level;
                    rays2[2 * ((i+j*width) * level + q) + 1].depth = level;

                    rays2[2 * ((i+j*width) * level + q)].hited = false;
                    rays2[2 * ((i+j*width) * level + q) + 1].hited = false;

                    rays2[2 * ((i+j*width) * level + q)].color = back_color;
                    rays2[2 * ((i+j*width) * level + q) + 1].color = back_color;
                }
            }
        }
    }
    __syncthreads();
}

__global__ void kernel2(int level, int width, int height, Ray* rays1, Ray* rays2, Ray* rays3, Ray* rays4, Ray* rays5, float3* framebuffer) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    if (level == 8) {
        for (int j = y; j < height; j += offset_y) {
            for (int i = x; i < width; i += offset_x) {
                for (int q = 0; q < level; q++) {
                    if (rays4[((i+j*width) * level + q)].hited)
                        rays4[((i+j*width) * level + q)].color = rays4[((i+j*width) * level + q)].color + rays5[2 * (((i+j*width) * level + q))].color * rays4[((i+j*width) * level + q)].material.albedo.z + rays5[2 * (((i+j*width) * level + q)) + 1].dir * rays4[((i+j*width) * level + q)].material.albedo.w;
                    if (rays4[((i+j*width) * level + q) + 1].hited)
                        rays4[((i+j*width) * level + q) + 1].color = rays4[((i+j*width) * level + q) + 1].color + rays5[2 * (((i+j*width) * level + q) + 1)].color * rays4[((i+j*width) * level + q) + 1].material.albedo.z + rays5[2 * (((i+j*width) * level + q) + 1) + 1].dir * rays4[((i+j*width) * level + q) + 1].material.albedo.w;
                }
            }
        }
        level = 4;
        __syncthreads();
    }

    if (level == 4) {
        for (int j = y; j < height; j += offset_y) {
            for (int i = x; i < width; i += offset_x) {
                for (int q = 0; q < level; q++) {
                    if (rays3[((i+j*width) * level + q)].hited)
                        rays3[((i+j*width) * level + q)].color = rays3[((i+j*width) * level + q)].color + rays4[2 * (((i+j*width) * level + q))].color * rays3[((i+j*width) * level + q)].material.albedo.z + rays4[2 * (((i+j*width) * level + q)) + 1].dir * rays3[((i+j*width) * level + q)].material.albedo.w;
                    if (rays3[((i+j*width) * level + q) + 1].hited)
                        rays3[((i+j*width) * level + q) + 1].color = rays3[((i+j*width) * level + q) + 1].color + rays4[2 * (((i+j*width) * level + q) + 1)].color * rays3[((i+j*width) * level + q) + 1].material.albedo.z + rays4[2 * (((i+j*width) * level + q) + 1) + 1].dir * rays3[((i+j*width) * level + q) + 1].material.albedo.w;
                }
            }
        }
        level = 2;
        __syncthreads();
    }

    if (level == 2) {
        for (int j = y; j < height; j += offset_y) {
            for (int i = x; i < width; i += offset_x) {
                for (int q = 0; q < level; q++) {
                    if (rays2[((i+j*width) * level + q)].hited)
                        rays2[((i+j*width) * level + q)].color = rays2[((i+j*width) * level + q)].color + rays3[2 * (((i+j*width) * level + q))].color * rays2[((i+j*width) * level + q)].material.albedo.z + rays3[2 * (((i+j*width) * level + q)) + 1].dir * rays2[((i+j*width) * level + q)].material.albedo.w;
                    if (rays2[((i+j*width) * level + q) + 1].hited)
                        rays2[((i+j*width) * level + q) + 1].color = rays2[((i+j*width) * level + q) + 1].color + rays3[2 * (((i+j*width) * level + q) + 1)].color * rays2[((i+j*width) * level + q) + 1].material.albedo.z + rays3[2 * (((i+j*width) * level + q) + 1) + 1].dir * rays2[((i+j*width) * level + q) + 1].material.albedo.w;
                }
            }
        }
        level = 1;
        __syncthreads();
    }

    if (level == 1) {
        for (int j = y; j < height; j += offset_y) {
            for (int i = x; i < width; i += offset_x) {
                for (int q = 0; q < level; q++) {
                    if (rays1[((i+j*width) * level + q)].hited)
                        rays1[((i+j*width) * level + q)].color = rays1[((i+j*width) * level + q)].color + rays2[2 * (((i+j*width) * level + q))].color * rays1[((i+j*width) * level + q)].material.albedo.z + rays2[2 * (((i+j*width) * level + q)) + 1].dir * rays1[((i+j*width) * level + q)].material.albedo.w;
                }
            }
        }
        __syncthreads();
    }

    for (int j = y; j < height; j += offset_y) {
        for (int i = x; i < width; i += offset_x) {
            framebuffer[i + j * width] = rays1[i + j * width].color;
        }
    }
}


void render1(int th, int depth, float3 back_color, Camera& cam, const Sphere* spheres, int ssize, const Tetrahedron* tetrahedrones, int tsize,
             const Cube* cubes, int csize, const Icosahedron* icosahedrones, int isize,
             const Light* lights, int lsize) {
    const int width    = 1024;
    const int height   = 768;
    
    float3* framebuffer = (float3*)malloc(sizeof(float3) * width * height);
    
    float3* framebuffer_gpu;
    
    Sphere* spheres_gpu;
    Tetrahedron* tetrahedrones_gpu;
    Cube* cubes_gpu;
    Icosahedron* icosahedrones_gpu;
    Light* lights_gpu;
    
    CSC(cudaMalloc((void**)&framebuffer_gpu, width * height * sizeof(float3)));
    
    CSC(cudaMalloc((void**)&spheres_gpu, ssize * sizeof(Sphere)));
    CSC(cudaMalloc((void**)&tetrahedrones_gpu, tsize * sizeof(Tetrahedron)));
    CSC(cudaMalloc((void**)&cubes_gpu, csize * sizeof(Cube)));
    CSC(cudaMalloc((void**)&icosahedrones_gpu, isize * sizeof(Icosahedron)));
    CSC(cudaMalloc((void**)&lights_gpu, lsize * sizeof(Light)));
    
    CSC(cudaMemcpy(spheres_gpu, spheres, ssize * sizeof(Sphere), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(tetrahedrones_gpu, tetrahedrones, tsize * sizeof(Tetrahedron), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(cubes_gpu, cubes, csize * sizeof(Cube), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(icosahedrones_gpu, icosahedrones, isize * sizeof(Icosahedron), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(lights_gpu, lights, lsize * sizeof(Light), cudaMemcpyHostToDevice));


    Ray* rays1;
    CSC(cudaMalloc((void**)&rays1, width * height * sizeof(Ray)));    
    
    Ray* rays2;
    CSC(cudaMalloc((void**)&rays2, 2 * width * height * sizeof(Ray)));
    
    Ray* rays3;
    if (depth >= 2)
        CSC(cudaMalloc((void**)&rays3, 4 * width * height * sizeof(Ray)));
    
    Ray* rays4;
    if (depth >= 4)
        CSC(cudaMalloc((void**)&rays4, 8 * width * height * sizeof(Ray)));
    
    Ray* rays5;
    if (depth == 8)
        CSC(cudaMalloc((void**)&rays5, 16 * width * height * sizeof(Ray)));

    kernel1<<<th, th>>>(back_color, cam, 1, rays1, rays2, width, height, framebuffer_gpu, spheres_gpu, ssize, tetrahedrones_gpu, tsize, cubes_gpu, csize, icosahedrones_gpu, isize, lights_gpu, lsize);
    if (depth >= 2)
        kernel1<<<th, th>>>(back_color, cam, 2, rays2, rays3, width, height, framebuffer_gpu, spheres_gpu, ssize, tetrahedrones_gpu, tsize, cubes_gpu, csize, icosahedrones_gpu, isize, lights_gpu, lsize);
    if (depth >= 4)
        kernel1<<<th, th>>>(back_color, cam, 4, rays3, rays4, width, height, framebuffer_gpu, spheres_gpu, ssize, tetrahedrones_gpu, tsize, cubes_gpu, csize, icosahedrones_gpu, isize, lights_gpu, lsize);
    if (depth == 8)
        kernel1<<<th, th>>>(back_color, cam, 8, rays4, rays5, width, height, framebuffer_gpu, spheres_gpu, ssize, tetrahedrones_gpu, tsize, cubes_gpu, csize, icosahedrones_gpu, isize, lights_gpu, lsize);
    
    
    kernel2<<<th, th>>>(depth, width, height, rays1, rays2, rays3, rays4, rays5, framebuffer_gpu);

    CSC(cudaMemcpy(framebuffer, framebuffer_gpu, width * height * sizeof(float3), cudaMemcpyDeviceToHost));

    std::ofstream ofs; // save the framebuffer to file
    
    char name[25];
    memset(name, '\0', sizeof(char) * 25);
    strcpy(name, "frames/out");
    char i[3] = {'\0', '\0', '\0'};
    itoa(cam.id, i, 10);
    strcat(name, i);
    strcat(name, ".ppm");
    
    printf("%s\n", name);
    
    ofs.open(name, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height*width; ++i) {
        float3 &c = framebuffer[i];
        float max = std::max(c.x, std::max(c.y, c.z));
        if (max > 1) c = c * (1. / max);
        ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i].x)));
        ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i].y)));
        ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i].z)));
    }
    ofs.close();
    cudaFree(framebuffer_gpu);
    cudaFree(spheres_gpu);
    cudaFree(tetrahedrones_gpu);
    cudaFree(cubes_gpu);
    cudaFree(icosahedrones_gpu);
    cudaFree(lights_gpu);
    
    cudaFree(rays1);
    cudaFree(rays2);
    if (depth >= 2)
        cudaFree(rays3);
    if (depth >= 4)
        cudaFree(rays4);
    if (depth >= 8)
        cudaFree(rays5);
}

__host__ __device__ float Distance(const float3& A, const float3& B) {
    return sqrtf((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y) + (A.z - B.z) * (A.z - B.z));
}

__host__ __device__ void LineMask(const float3 A, const float3 B, int n, float r, Sphere* spheres, int ssize, Light* lights, int lsize) {
    Material white(1.0, make_float4(0.6,  0.3, 0.1, 0.0), make_float3(1, 1, 1),   0.);
    float d = Distance(A, B);
    float3 vec = normalize((B - A));
    float3 step = vec * (d / n);
    
    for (float i = 0; i < n; i++) {
        lights[lsize + (int)i] = Light(A + step * i, 25);
        spheres[ssize + (int)i] = Sphere(A + step * i, r, white);
    }
}

__global__ void kernel3(const Icosahedron i, int n, float r, Sphere* spheres, int ssize, Light* lights, int lsize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int offset_x = blockDim.x * gridDim.x;
        
    for (int j = x; j < 20; j += offset_x) {
        LineMask(i.VerMas[(int)(i.IndMas[j].x)], i.VerMas[(int)(i.IndMas[j].y)], n, r, spheres, ssize + j * n * 3, lights, lsize + j * n * 3);
        LineMask(i.VerMas[(int)(i.IndMas[j].x)], i.VerMas[(int)(i.IndMas[j].z)], n, r, spheres, ssize + j * n * 3 + n, lights, lsize + j * n * 3 + n);
        LineMask(i.VerMas[(int)(i.IndMas[j].z)], i.VerMas[(int)(i.IndMas[j].y)], n, r, spheres, ssize + j * n * 3 + 2 * n, lights, lsize + j * n * 3 + 2 * n);
    }
}

void TetrahedronMask(const Tetrahedron& t, int n, float r, Sphere* spheres, int &ssize, Light* lights, int &lsize) {
    LineMask(t.S, t.A, n, r, spheres, ssize, lights, lsize);
    LineMask(t.S, t.B, n, r, spheres, ssize + n, lights, lsize + n);
    LineMask(t.S, t.C, n, r, spheres, ssize + 2 * n, lights, lsize + 2 * n);
    LineMask(t.A, t.B, n, r, spheres, ssize + 3 * n, lights, lsize + 3 * n);
    LineMask(t.B, t.C, n, r, spheres, ssize + 4 * n, lights, lsize + 4 * n);
    LineMask(t.C, t.A, n, r, spheres, ssize + 5 * n, lights, lsize + 5 * n);
    
    ssize = ssize + 6 * n;
    lsize = lsize + 6 * n;
}

void CubeMask(const Cube& c, int n, float r, Sphere* spheres, int &ssize, Light* lights, int &lsize) {
    LineMask(c.A, c.B, n, r, spheres, ssize, lights, lsize);
    LineMask(c.B, c.C, n, r, spheres, ssize + n, lights, lsize + 1 * n);
    LineMask(c.C, c.D, n, r, spheres, ssize + 2 * n, lights, lsize + 2 * n);
    LineMask(c.D, c.A, n, r, spheres, ssize + 3 * n, lights, lsize + 3 * n);

    LineMask(c.A1, c.B1, n, r, spheres, ssize + 4 * n, lights, lsize + 4 * n);
    LineMask(c.B1, c.C1, n, r, spheres, ssize + 5 * n, lights, lsize + 5 * n);
    LineMask(c.C1, c.D1, n, r, spheres, ssize + 6 * n, lights, lsize + 6 * n);
    LineMask(c.D1, c.A1, n, r, spheres, ssize + 7 * n, lights, lsize + 7 * n);
    
    LineMask(c.A1, c.A, n, r, spheres, ssize + 8 * n, lights, lsize + 8 * n);
    LineMask(c.B1, c.B, n, r, spheres, ssize + 9 * n, lights, lsize + 9 * n);
    LineMask(c.C1, c.C, n, r, spheres, ssize + 10 * n, lights, lsize + 10 * n);
    LineMask(c.D1, c.D, n, r, spheres, ssize + 11 * n, lights, lsize + 11 * n);
    
    ssize = ssize + 12 * n;
    lsize = lsize + 12 * n;
}

void IcosahedronMask(const Icosahedron& i, int n, float r, Sphere* spheres, int &ssize, Light* lights, int &lsize) {
    Sphere* spheres_gpu;
    Light* lights_gpu;
    
    CSC(cudaMalloc((void**)&spheres_gpu, (ssize + n * 20 * 3) * sizeof(Sphere)));
    CSC(cudaMalloc((void**)&lights_gpu, (lsize + n * 20 * 3) * sizeof(Light)));
    CSC(cudaMemcpy(spheres_gpu, spheres, (ssize) * sizeof(Sphere), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(lights_gpu, lights, (lsize) * sizeof(Light), cudaMemcpyHostToDevice));

    kernel3<<<2, 2>>>(i, n, r, spheres_gpu, ssize, lights_gpu, lsize);
    
    CSC(cudaMemcpy(spheres, spheres_gpu, (ssize + n * 20 * 3) * sizeof(Sphere), cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(lights, lights_gpu, (lsize + n * 20 * 3) * sizeof(Light), cudaMemcpyDeviceToHost));
        
    cudaFree(spheres_gpu);
    cudaFree(lights_gpu);
    
    ssize = ssize + n * 20 * 3;
    lsize = lsize + n * 20 * 3;
}


int main() {// x, y, по оси z направлены лучи зрения

    Material      ivory(1.0, make_float4(0.6,  0.3, 0.1, 0.0), make_float3(0.4, 0.4, 0.3),   50.);
    Material      glass(1.5, make_float4(0.0,  0.5, 0.1, 0.8), make_float3(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0.3, 0.1, 0.1),   10.);
    Material blue(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0.1, 0.1, 0.3),   10.);
    Material green(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0.1, 0.3, 0.1),   10.);
    Material black(1.0, make_float4(0.9,  0.1, 0.0, 0.0), make_float3(0., 0., 0.),   10.);
    Material white(1.0, make_float4(0.6,  0.3, 0.1, 0.0), make_float3(1, 1, 1),   0.);
    Material     mirror(1.0, make_float4(0.0, 10.0, 0.8, 0.0), make_float3(1.0, 1.0, 1.0), 1425.);

    int ssize = 0;//    number of spheres
    scanf("%d", &ssize);
    Sphere* spheres = (Sphere*)malloc(sizeof(Sphere) * ssize);
    for (int i = 0; i < ssize; i++) {
        float3 coord;
        scanf("%f%f%f", &coord.x, &coord.y, &coord.z);
        float radius;
        scanf("%f", &radius);
        int color;
        scanf("%d", &color);
        // spheres[i] = Sphere(make_float3(-1,    -2,   -12), 1.5, red_rubber);
        if (color == 1)
            spheres[i] = Sphere(coord, radius, red_rubber);
        else if (color == 2)
            spheres[i] = Sphere(coord, radius, ivory);
        else if (color == 3)
            spheres[i] = Sphere(coord, radius, black);
        else if (color == 4)
            spheres[i] = Sphere(coord, radius, white);
        else if (color == 5)
            spheres[i] = Sphere(coord, radius, mirror);
        else if (color == 6)
            spheres[i] = Sphere(coord, radius, blue);
        else if (color == 7)
            spheres[i] = Sphere(coord, radius, green);
    }

    int tsize = 0;
    scanf("%d", &tsize);
    Tetrahedron* tetrahedrones = (Tetrahedron*)malloc(sizeof(Tetrahedron) * tsize);
    for (int i = 0; i < tsize; i++) {
        float3 coord;
        scanf("%f%f%f", &coord.x, &coord.y, &coord.z);
        float radius;
        scanf("%f", &radius);
        int color;
        scanf("%d", &color);
        // tetrahedrones[0] = Tetrahedron(make_float3(-5,    -0.,   -14), 3, red_rubber);
        if (color == 1)
            tetrahedrones[i] = Tetrahedron(coord, radius, red_rubber);
        else if (color == 2)
            tetrahedrones[i] = Tetrahedron(coord, radius, ivory);
        else if (color == 3)
            tetrahedrones[i] = Tetrahedron(coord, radius, black);
        else if (color == 4)
            tetrahedrones[i] = Tetrahedron(coord, radius, white);
        else if (color == 5)
            tetrahedrones[i] = Tetrahedron(coord, radius, mirror);
        else if (color == 6)
            tetrahedrones[i] = Tetrahedron(coord, radius, blue);
        else if (color == 7)
            tetrahedrones[i] = Tetrahedron(coord, radius, green);
    }
    
    int csize = 0;
    scanf("%d", &csize);
    Cube* cubes = (Cube*)malloc(sizeof(Cube) * csize);
    for (int i = 0; i < csize; i++) {
        float3 coord;
        scanf("%f%f%f", &coord.x, &coord.y, &coord.z);
        float radius;
        scanf("%f", &radius);
        int color;
        scanf("%d", &color);
        // cubes[0] = Cube(make_float3(.0, -0.5, -10), 2, red_rubber);
        if (color == 1)
            cubes[i] = Cube(coord, radius, red_rubber);
        else if (color == 2)
            cubes[i] = Cube(coord, radius, ivory);
        else if (color == 3)
            cubes[i] = Cube(coord, radius, black);
        else if (color == 4)
            cubes[i] = Cube(coord, radius, white);
        else if (color == 5)
            cubes[i] = Cube(coord, radius, mirror);
        else if (color == 6)
            cubes[i] = Cube(coord, radius, blue);
        else if (color == 7)
            cubes[i] = Cube(coord, radius, green);
    }

    int isize = 0;
    scanf("%d", &isize);
    Icosahedron* icosahedrones = (Icosahedron*)malloc(sizeof(Icosahedron) * isize);
    for (int i = 0; i < isize; i++) {
        float3 coord;
        scanf("%f%f%f", &coord.x, &coord.y, &coord.z);
        float radius;
        scanf("%f", &radius);
        int color;
        scanf("%d", &color);
        // icosahedrones[0] = Icosahedron(make_float3(0,    3.5,   -14), 3, ivory);
        if (color == 1)
            icosahedrones[i] = Icosahedron(coord, radius, red_rubber);
        else if (color == 2)
            icosahedrones[i] = Icosahedron(coord, radius, ivory);
        else if (color == 3)
            icosahedrones[i] = Icosahedron(coord, radius, black);
        else if (color == 4)
            icosahedrones[i] = Icosahedron(coord, radius, white);
        else if (color == 5)
            icosahedrones[i] = Icosahedron(coord, radius, mirror);
        else if (color == 6)
            icosahedrones[i] = Icosahedron(coord, radius, blue);
        else if (color == 7)
            icosahedrones[i] = Icosahedron(coord, radius, green);
    }
    
    int lsize = 0;
    scanf("%d", &lsize);
    Light* lights = (Light*)malloc(sizeof(Light) * lsize);
    for (int i = 0; i < lsize; i++) {
        float3 coord;
        scanf("%f%f%f", &coord.x, &coord.y, &coord.z);
        float intensity;
        scanf("%f", &intensity);
        // lights[0] = Light(make_float3(0, 30,  20), 3.5);
        lights[i] = Light(coord, intensity);
    }
    
    int code = 0, n = 0;
    scanf("%d", &code);
    if (code) {
        scanf("%d", &code);
        if (code) {
            scanf("%d", &n);
            float size = 0.04;
            scanf("%f", &size);
            spheres = (Sphere*)realloc(spheres, sizeof(Sphere) * (ssize + 12 * n) * csize);
            lights = (Light*)realloc(lights, sizeof(Light) * (lsize + 12 * n) * csize);
            for (int i = 0; i < csize; i++)
                CubeMask(cubes[i], n, size, spheres, ssize, lights, lsize);
        }

        scanf("%d", &code);
        if (code) {
            scanf("%d", &n);
            float size = 0.04;
            scanf("%f", &size);
            spheres = (Sphere*)realloc(spheres, sizeof(Sphere) * (ssize + n * 20 * 3) * isize);
            lights = (Light*)realloc(lights, sizeof(Light) * (lsize + n * 20 * 3) * isize);
            for (int i = 0; i < isize; i++)
                IcosahedronMask(icosahedrones[i], n, size, spheres, ssize, lights, lsize);
        }

        scanf("%d", &code);
        if (code) {
            scanf("%d", &n);
            float size = 0.04;
            scanf("%f", &size);
            spheres = (Sphere*)realloc(spheres, sizeof(Sphere) * (ssize + 6 * n) * tsize);
            lights = (Light*)realloc(lights, sizeof(Light) * (lsize + 6 * n) * tsize);
            for (int i = 0; i < tsize; i++)
                TetrahedronMask(tetrahedrones[i], n, size, spheres, ssize, lights, lsize);
        }
    }

    int n_frames = 0;
    float rc0, prc, z0c, pzc, Arc, Azc, wrc, wzc, wphic, phi0c, rn0, wrn, wzn, wphin, prn, Arn, Azn, z0n, pzn, phi0n;
    scanf("%d", &n_frames);

    float step = 2 * M_PI / n_frames;
    float t = 0, rc, phic, zc, rn, phin, zn;
    float field_of_view = M_PI / 3.;
    const int   pixel_width    = 1024;
    const int   pixel_height   = 768;
    
    scanf("%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f", &rc0, &z0c, &phi0c, &Arc, &Azc, &wrc, &wzc, &wphic, &prc, &pzc, \
                                                      &rn0, &z0n, &phi0n, &Arn, &Azn, &wrn, &wzn, &wphin, &prn, &pzn);
    int th = 32, depth = 0;
    float3 back_color = make_float3(0, 0, 0);
    scanf("%d%d%f%f%f", &th, &depth, &back_color.x, &back_color.y, &back_color.z);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < n_frames; i++) {
        rc = rc0 + Arc * sin(wrc * t + prc);
        zc = z0c + Azc * sin(wzc * t + pzc);
        phic = phi0c + wphic * t;
        
        rn = rn0 + Arn * sin(wrn * t + prn);
        zn = z0n + Azn * sin(wzn * t + pzn);
        phin = phi0n + wphin * t;
        
        Camera cam = Camera(i, rc, phic, zc, rn, phin, zn, pixel_width, pixel_height, field_of_view);
        
        render1(th, depth, back_color, cam, spheres, ssize, tetrahedrones, tsize, cubes, csize, icosahedrones, isize, lights, lsize);
        
        t += step;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time = %.10f\n", milliseconds);

    free(spheres);
    free(tetrahedrones);
    free(cubes);
    free(icosahedrones);
    free(lights);

    std::cout << "done" << std::endl;

    return 0;
}
