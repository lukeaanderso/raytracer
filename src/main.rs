use rustmath::Matrix;
use image::{ImageBuffer, Rgb};
use std::path::Path;
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// 3D Vector representation
#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn normalize(&self) -> Vec3 {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Vec3 {
            x: self.x / length,
            y: self.y / length,
            z: self.z / length,
        }
    }

    fn subtract(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn scale(&self, t: f64) -> Vec3 {
        Vec3 {
            x: self.x * t,
            y: self.y * t,
            z: self.z * t,
        }
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn multiply(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

// Ray representation
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    fn point_at(&self, t: f64) -> Vec3 {
        Vec3 {
            x: self.origin.x + self.direction.x * t,
            y: self.origin.y + self.direction.y * t,
            z: self.origin.z + self.direction.z * t,
        }
    }
}

// Object trait for different shapes
use std::any::Any;

trait Object: Send + Sync {
    fn intersect(&self, ray: &Ray) -> Option<f64>;
    fn normal_at(&self, point: &Vec3) -> Vec3;
    fn box_clone(&self) -> Box<dyn Object>;
    fn as_any(&self) -> &dyn Any;
    fn center(&self) -> Option<Vec3> {
        None
    }
}

impl Clone for Box<dyn Object> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

// Sphere representation
#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f64,
}

impl Sphere {
    fn new(center: Vec3, radius: f64) -> Self {
        Sphere { center, radius }
    }
}

impl Object for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let oc = ray.origin.subtract(&self.center);
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            None
        } else {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            if t > 0.0 {
                Some(t)
            } else {
                let t = (-b + discriminant.sqrt()) / (2.0 * a);
                if t > 0.0 { Some(t) } else { None }
            }
        }
    }

    fn normal_at(&self, point: &Vec3) -> Vec3 {
        point.subtract(&self.center).normalize()
    }

    fn box_clone(&self) -> Box<dyn Object> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn center(&self) -> Option<Vec3> {
        Some(self.center)
    }
}

// Material types
#[derive(Clone, Copy)]
enum MaterialType {
    Diffuse,
    Metal { fuzz: f64 },
    Glass { refractive_index: f64 },
}

// Material properties
#[derive(Clone)]
struct Material {
    material_type: MaterialType,
    color: Vec3,
    ambient: f64,
    diffuse: f64,
}

impl Material {
    fn metal(color: Vec3, fuzz: f64) -> Self {
        Material {
            material_type: MaterialType::Metal { fuzz: fuzz.clamp(0.0, 1.0) },
            color,
            ambient: 0.1,
            diffuse: 0.6,
        }
    }

    fn glass(refractive_index: f64) -> Self {
        Material {
            material_type: MaterialType::Glass { refractive_index },
            color: Vec3::new(0.95, 0.95, 1.0),
            ambient: 0.1,
            diffuse: 0.1,
        }
    }

    fn diffuse(color: Vec3) -> Self {
        Material {
            material_type: MaterialType::Diffuse,
            color,
            ambient: 0.2,
            diffuse: 0.8,
        }
    }

    fn random_diffuse(rng: &mut impl Rng) -> Self {
        Material {
            material_type: MaterialType::Diffuse,
            color: Vec3::new(
                rng.gen_range(0.3..0.9),
                rng.gen_range(0.3..0.9),
                rng.gen_range(0.3..0.9),
            ),
            ambient: 0.1,
            diffuse: 0.9,
        }
    }

    fn random_metal(rng: &mut impl Rng) -> Self {
        Material {
            material_type: MaterialType::Metal {
                fuzz: rng.gen_range(0.0..0.2),
            },
            color: Vec3::new(
                rng.gen_range(0.6..0.9),
                rng.gen_range(0.6..0.9),
                rng.gen_range(0.6..0.9),
            ),
            ambient: 0.1,
            diffuse: 0.5,
        }
    }
}

fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = (-uv.dot(n)).min(1.0);
    let r_out_perp = (uv.add(&n.scale(cos_theta))).scale(etai_over_etat);
    let r_out_parallel = n.scale(-((1.0 - r_out_perp.dot(&r_out_perp)).abs().sqrt()));
    r_out_perp.add(&r_out_parallel)
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Use Schlick's approximation for reflectance
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn gamma_correct(color: Vec3) -> Vec3 {
    Vec3 {
        x: color.x.clamp(0.0, 1.0).powf(1.0/2.2),
        y: color.y.clamp(0.0, 1.0).powf(1.0/2.2),
        z: color.z.clamp(0.0, 1.0).powf(1.0/2.2),
    }
}

fn color_ray(ray: &Ray, objects: &[SceneObject], depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut closest_hit: Option<(f64, &SceneObject)> = None;

    // Find closest intersection
    for object in objects {
        if let Some(t) = object.shape.intersect(ray) {
            match closest_hit {
                None => closest_hit = Some((t, object)),
                Some((closest_t, _)) if t < closest_t => closest_hit = Some((t, object)),
                _ => {}
            }
        }
    }

    if let Some((t, object)) = closest_hit {
        let hit_point = ray.point_at(t);
        let normal = object.shape.normal_at(&hit_point);
        let mut rng = rand::thread_rng();
        
        match object.material.material_type {
            MaterialType::Diffuse => {
                // Light direction (from slightly above and in front)
                let light_dir = Vec3::new(0.0, 1.0, -0.5).normalize();
                
                // Check for shadows
                let in_shadow = is_in_shadow(&hit_point, &light_dir, objects);
                let light_intensity = if in_shadow {
                    0.0
                } else {
                    normal.dot(&light_dir).max(0.0)
                };
                
                let ambient = object.material.color.scale(object.material.ambient);
                let diffuse = object.material.color.scale(light_intensity * object.material.diffuse);
                ambient.add(&diffuse)
            },
            MaterialType::Metal { fuzz } => {
                let reflected = ray.direction.subtract(&normal.scale(2.0 * ray.direction.dot(&normal)));
                // Add fuzz to the reflection
                let scattered = reflected.add(&Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0)
                ).scale(fuzz));
                
                let scattered_ray = Ray::new(hit_point.add(&scattered.scale(0.001)), scattered.normalize());
                let reflected_color = color_ray(&scattered_ray, objects, depth - 1);
                Vec3 {
                    x: reflected_color.x * object.material.color.x,
                    y: reflected_color.y * object.material.color.y,
                    z: reflected_color.z * object.material.color.z,
                }
            },
            MaterialType::Glass { refractive_index } => {
                let unit_direction = ray.direction.normalize();
                let outward_normal = if ray.direction.dot(&normal) > 0.0 {
                    normal.scale(-1.0)
                } else {
                    normal
                };
                
                let ni_over_nt = if ray.direction.dot(&normal) > 0.0 {
                    refractive_index
                } else {
                    1.0 / refractive_index
                };

                let cos_theta = (-unit_direction.dot(&outward_normal)).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = ni_over_nt * sin_theta > 1.0;
                let fresnel = reflectance(cos_theta, refractive_index);
                let will_reflect = fresnel > rng.gen::<f64>();

                let scattered_direction = if cannot_refract || will_reflect {
                    // Reflect
                    unit_direction.subtract(&outward_normal.scale(2.0 * unit_direction.dot(&outward_normal)))
                } else {
                    // Refract
                    refract(&unit_direction, &outward_normal, ni_over_nt)
                };

                let scattered_ray = Ray::new(hit_point.add(&scattered_direction.scale(0.001)), scattered_direction);
                let attenuation = object.material.color;
                color_ray(&scattered_ray, objects, depth - 1).multiply(&attenuation)
            }
        }
    } else {
        // Background color (sky)
        Vec3::new(0.5, 0.7, 1.0)
    }
}

// Scene object combining shape and material
#[derive(Clone)]
struct SceneObject {
    shape: Box<dyn Object>,
    material: Material,
}

// Camera orientation
#[derive(Clone)]
struct Camera {
    position: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

impl Camera {
    fn look_at(position: Vec3, target: Vec3, world_up: Vec3) -> Self {
        let forward = target.subtract(&position).normalize();
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward).normalize();

        Camera {
            position,
            forward,
            up,
            right,
        }
    }

    fn get_ray_direction(&self, u: f64, v: f64, viewport_width: f64, viewport_height: f64, focal_length: f64) -> Vec3 {
        let horizontal = self.right.scale(viewport_width);
        let vertical = self.up.scale(viewport_height);
        
        // Calculate the point on the viewport
        let viewport_center = self.position.add(&self.forward.scale(focal_length));
        let viewport_u = horizontal.scale(u - 0.5);
        let viewport_v = vertical.scale(v - 0.5);
        
        viewport_center
            .add(&viewport_u)
            .add(&viewport_v)
            .subtract(&self.position)
            .normalize()
    }
}

fn is_in_shadow(hit_point: &Vec3, light_dir: &Vec3, objects: &[SceneObject]) -> bool {
    let shadow_ray = Ray::new(
        // Start slightly above surface to avoid self-intersection
        hit_point.add(&light_dir.scale(0.001)),
        *light_dir,
    );

    // Check if any object blocks the light
    for object in objects {
        if object.shape.intersect(&shadow_ray).is_some() {
            return true;
        }
    }
    false
}

// Scene configuration
#[derive(Clone)]
struct Scene {
    width: u32,
    height: u32,
    camera: Camera,
    viewport_height: f64,
    viewport_width: f64,
    focal_length: f64,
    objects: Vec<SceneObject>,
    samples_per_pixel: u32,  // For anti-aliasing
}

impl Scene {
    fn generate_random_spheres(rng: &mut impl Rng) -> Vec<SceneObject> {
        let mut spheres: Vec<SceneObject> = Vec::new();
        
        // Add 15 random small spheres
        for _ in 0..15 {
            let radius = 0.12; // Keep them all the same small size
            
            // Try to find a non-overlapping position
            let mut attempts = 0;
            let max_attempts = 100;
            
            'position_search: loop {
                // Generate random position within reasonable bounds
                let pos = Vec3::new(
                    rng.gen_range(-2.0..2.0),
                    radius,  // Keep them on the ground
                    rng.gen_range(-1.5..1.5),
                );
                
                // Check distance from main spheres (hardcoded positions)
                let main_positions = [
                    (Vec3::new(0.0, 0.7, 0.8), 0.9),   // glass
                    (Vec3::new(-1.4, 0.45, 0.2), 0.6), // steel
                    (Vec3::new(0.8, 0.25, -0.6), 0.25), // small glass
                    (Vec3::new(1.4, 0.35, 0.2), 0.45),  // copper
                    (Vec3::new(0.0, 0.25, -1.0), 0.35), // orange
                    (Vec3::new(-0.7, 0.15, -0.5), 0.15), // blue
                ];
                
                // Check distance from existing random spheres
                let mut too_close = false;
                for existing_sphere in &spheres {
                    if let Some(sphere) = existing_sphere.shape.as_any().downcast_ref::<Sphere>() {
                        let dist = pos.subtract(&sphere.center).length();
                        if dist < (radius + 0.12) * 1.2 { // Add 20% spacing
                            too_close = true;
                            break;
                        }
                    }
                }
                
                if too_close {
                    attempts += 1;
                    if attempts >= max_attempts {
                        break;
                    }
                    continue;
                }
                
                // Check distance from main spheres
                for (center, main_radius) in main_positions.iter() {
                    let dist = pos.subtract(center).length();
                    if dist < (radius + main_radius) * 1.2 { // Add 20% spacing
                        attempts += 1;
                        if attempts >= max_attempts {
                            break 'position_search;
                        }
                        continue 'position_search;
                    }
                }
                
                // Position is good, create the sphere
                let material = if rng.gen_bool(0.7) {
                    Material::random_diffuse(rng)
                } else if rng.gen_bool(0.7) {
                    Material::random_metal(rng)
                } else {
                    Material::glass(1.52)
                };
                
                spheres.push(SceneObject {
                    shape: Box::new(Sphere::new(pos, radius)),
                    material,
                });
                
                break;
            }
        }
        
        spheres
    }

    fn new(width: u32, height: u32, samples_per_pixel: u32) -> Self {
        let aspect_ratio = width as f64 / height as f64;
        let viewport_height = 2.0;
        let mut rng = rand::thread_rng();
        
        let camera = Camera::look_at(
            Vec3::new(0.0, 1.2, -4.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        let mut objects = vec![
            // Large glass sphere - center back
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(0.0, 0.7, 0.8), 0.9)),
                material: Material::glass(1.52),
            },
            // Steel sphere - left
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(-1.4, 0.45, 0.2), 0.6)),
                material: Material::metal(Vec3::new(0.7, 0.7, 0.75), 0.05),
            },
            // Small glass sphere - front right
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(0.8, 0.25, -0.6), 0.25)),
                material: Material::glass(1.52),
            },
            // Copper sphere - right
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(1.4, 0.35, 0.2), 0.45)),
                material: Material::metal(Vec3::new(0.722, 0.451, 0.20), 0.1),
            },
            // Orange sphere - front center
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(0.0, 0.25, -1.0), 0.35)),
                material: Material::diffuse(Vec3::new(1.0, 0.5, 0.0)),
            },
            // Blue sphere - front left
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(-0.7, 0.15, -0.5), 0.15)),
                material: Material::diffuse(Vec3::new(0.1, 0.2, 0.6)),
            },
            // Ground plane (large sphere below)
            SceneObject {
                shape: Box::new(Sphere::new(Vec3::new(0.0, -200.0, 0.0), 200.0)),
                material: Material::diffuse(Vec3::new(0.2, 0.2, 0.2)),
            },
        ];
        
        // Add random small spheres
        objects.extend(Self::generate_random_spheres(&mut rng));

        Scene {
            width,
            height,
            camera,
            viewport_height,
            viewport_width: viewport_height * aspect_ratio,
            focal_length: 1.0,
            objects,
            samples_per_pixel,
        }
    }

    fn get_pixel_color(&self, i: u32, j: u32) -> Vec3 {
        let mut rng = rand::thread_rng();
        let mut color = Vec3::new(0.0, 0.0, 0.0);

        // Anti-aliasing: Take multiple samples per pixel
        for _ in 0..self.samples_per_pixel {
            // Add random offset within the pixel
            let u = ((i as f64) + rng.gen::<f64>()) / (self.width as f64 - 1.0);
            let v = ((j as f64) + rng.gen::<f64>()) / (self.height as f64 - 1.0);

            let direction = self.camera.get_ray_direction(
                u, v,
                self.viewport_width,
                self.viewport_height,
                self.focal_length
            );

            let ray = Ray::new(self.camera.position, direction);
            color = color.add(&color_ray(&ray, &self.objects, 5));
        }

        // Average the samples and apply gamma correction
        gamma_correct(color.scale(1.0 / self.samples_per_pixel as f64))
    }

    fn render(&self) -> (Matrix, Matrix, Matrix) {
        let height = self.height as usize;
        let width = self.width as usize;
        let start_time = Instant::now();

        // Create vectors to store our color data
        let mut pixels = vec![(0.0, 0.0, 0.0); width * height];
        let total_pixels = width * height;
        
        // Create atomic counter for progress
        let counter = Arc::new(AtomicUsize::new(0));
        let last_print = Arc::new(AtomicUsize::new(0));

        println!("Starting render at {}x{} with {} samples per pixel...", width, height, self.samples_per_pixel);
        print!("Progress: 0%");
        io::stdout().flush().unwrap();

        // Process pixels in parallel
        pixels.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            let i = (idx % width) as u32;
            let j = (idx / width) as u32;
            let color = self.get_pixel_color(i, j);
            *pixel = (color.x, color.y, color.z);

            // Update progress atomically
            let completed = counter.fetch_add(1, Ordering::Relaxed);
            let percent = (completed * 100) / total_pixels;
            let last = last_print.load(Ordering::Relaxed);
            
            if percent > last && last_print.compare_exchange(last, percent, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                let elapsed = start_time.elapsed();
                let eta = if percent > 0 {
                    elapsed.mul_f64(100.0 / percent as f64) - elapsed
                } else {
                    elapsed
                };
                print!("\rProgress: {}% (ETA: {:.1}s)    ", 
                    percent,
                    eta.as_secs_f64()
                );
                io::stdout().flush().unwrap();
            }
        });

        println!("\nRender completed in {:.1}s", start_time.elapsed().as_secs_f64());

        // Create the result matrices
        let mut r_data = vec![vec![0.0; width]; height];
        let mut g_data = vec![vec![0.0; width]; height];
        let mut b_data = vec![vec![0.0; width]; height];

        // Fill the matrices
        for (idx, (r, g, b)) in pixels.iter().enumerate() {
            let i = idx % width;
            let j = idx / width;
            r_data[j][i] = *r;
            g_data[j][i] = *g;
            b_data[j][i] = *b;
        }

        (
            Matrix::from_vec(r_data),
            Matrix::from_vec(g_data),
            Matrix::from_vec(b_data),
        )
    }
}

fn matrices_to_png(r: &Matrix, g: &Matrix, b: &Matrix, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Verify matrices are the same size
    let (height, width) = r.shape();
    assert_eq!(g.shape(), (height, width), "Green channel matrix has different dimensions");
    assert_eq!(b.shape(), (height, width), "Blue channel matrix has different dimensions");

    // Create a new RGB image
    let mut img = ImageBuffer::new(width as u32, height as u32);

    // Fill the image with RGB values from matrices
    for y in 0..height {
        for x in 0..width {
            // Get values from matrices and convert to u8 (0-255)
            let r_val = (r.get(y, x).unwrap().clamp(0.0, 1.0) * 255.0) as u8;
            let g_val = (g.get(y, x).unwrap().clamp(0.0, 1.0) * 255.0) as u8;
            let b_val = (b.get(y, x).unwrap().clamp(0.0, 1.0) * 255.0) as u8;

            // Set the pixel - flip y coordinate to correct image orientation
            img.put_pixel(x as u32, (height - 1 - y) as u32, Rgb([r_val, g_val, b_val]));
        }
    }

    // Save the image
    img.save(Path::new(output_path))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Double the resolution
    let width = 1600;
    let height = 1200;
    let samples_per_pixel = 1024;
    
    let scene = Scene::new(width, height, samples_per_pixel);
    let (r, g, b) = scene.render();
    matrices_to_png(&r, &g, &b, "sphere.png")?;
    Ok(())
}
