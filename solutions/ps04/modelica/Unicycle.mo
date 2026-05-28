model Unicycle
  parameter Real v_nom = 2.0;
  parameter Real omega_turn = 0.4;
  parameter Real T_lobe = 2*3.141592653589793/omega_turn;
  parameter Real T_fig8 = 2*T_lobe;

  Real x(start=0.0);
  Real y(start=0.0);
  Real theta(start=0.0);
  Real omega_ref;

equation
  // For the required 35 s simulation horizon, the figure-eight command only
  // needs three sign segments and avoids mod(time, T_fig8), which the current
  // Rumoca prepared runtime rejects during simulation.
  omega_ref =
    if time < T_lobe then omega_turn
    elseif time < T_fig8 then -omega_turn
    else omega_turn;

  der(x) = v_nom * cos(theta);
  der(y) = v_nom * sin(theta);
  der(theta) = omega_ref;

end Unicycle;
