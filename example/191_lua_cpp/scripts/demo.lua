print("Initial caption from C++:", get_caption())

-- UTF-8 test: emoji + non-Latin text
set_caption("Fractal 🌈 — 你好 from Lua")

local w, h = 800, 480
local img = Image(w, h)

-- Background colour
img:clear(Colour(0.02, 0.02, 0.08))

-- Simple math-based pattern: kind of fake "fractal-ish" swirl.
for y = 0, h - 1 do
    local ny = (y - h / 2) / (h / 2)
    for x = 0, w - 1 do
        local nx = (x - w / 2) / (w / 2)

        local r = math.sqrt(nx * nx + ny * ny)
        local angle = math.atan(ny, nx)

        local v = math.sin(10.0 * r + 5.0 * angle)
        local v2 = math.cos(8.0 * r - 4.0 * angle)

        local cr = 0.5 + 0.5 * v
        local cg = 0.3 + 0.5 * v2
        local cb = 0.6 + 0.4 * (v * v2)

        img:set_pixel(x, y, Colour(cr, cg, cb))
    end
end

-- Save with enum parameter (mode affects brightness slightly in C++).
img:save("output.ppm", BlendMode.Add)

-- Demonstrate C++ std::vector<int> exposed as IntVector
local v = make_vector()
print("C++ IntVector size:", v:size())
for i = 0, v:size() - 1 do
    print("v[" .. i .. "] = " .. v:get(i))
end

print("Final caption from Lua:", get_caption())
print("Image written to output.ppm (P6 binary PPM)")
