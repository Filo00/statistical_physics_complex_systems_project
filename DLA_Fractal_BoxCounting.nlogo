; DLA Fractal Aggregation + Fractal Dimension Estimation (NetLogo)

globals [
  sticked-count
  box-sizes
  fractal-dimension
]

turtles-own [
  stuck?
]

to setup
  clear-all
  set-default-shape turtles "circle"
  create-turtles 1 [
    setxy 0 0
    set stuck? true
    set color black
  ]
  set sticked-count 1
  reset-ticks
end

to go
  if sticked-count >= 1000 [ stop ]
  create-turtles 1 [
    setxy random-xcor random-ycor
    set stuck? false
    set color blue
  ]
  ask turtles with [not stuck?] [
    right random 360
    forward 1
    if any? turtles-on neighbors with [stuck?] [
      set stuck? true
      set color black
      set sticked-count sticked-count + 1
    ]
    if not can-move? 1 [ die ]
  ]
  tick
end

to box-counting
  let positions [list pxcor pycor] of turtles with [stuck?]
  if empty? positions [ print "Nessuna particella attaccata." stop ]

  set box-sizes [1 2 4 8 16]
  let log-box-size []
  let log-nboxes []

  foreach box-sizes [ boxsize ->
    let boxes []
    ask turtles with [stuck?] [
      let bx floor (pxcor / boxsize)
      let by floor (pycor / boxsize)
      let box (word bx "," by)
      if not member? box boxes [
        set boxes lput box boxes
      ]
    ]
    set log-box-size lput (ln (1.0 / boxsize)) log-box-size
    set log-nboxes lput (ln length boxes) log-nboxes
  ]

  let slope (linear-regression-slope log-box-size log-nboxes)
  set fractal-dimension slope
  print (word "Stima della dimensione frattale ≈ " slope)
end

to-report linear-regression-slope [xs ys]
  let n length xs
  let xbar mean xs
  let ybar mean ys
  let num sum (map [i -> (item i xs - xbar) * (item i ys - ybar)] (range n))
  let den sum (map [i -> (item i xs - xbar) ^ 2] (range n))
  report num / den
end
