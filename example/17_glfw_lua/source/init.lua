a = 3 + 11 + math.sin(332.2)

function AddStuff(a, b)
  print("[LUA] AddStuff("..a..","..b..")\n")
  return a + b
end

function DoAThing(a,b)
  print("[LUA] DoAThing("..a..","..b..")\n")
  c = HostFunction(a+10,b*3)
  return c
end