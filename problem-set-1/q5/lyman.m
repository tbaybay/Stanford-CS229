% Estimate the light spectrum of quasars
% Quasars are luminous distant galactic nuclei
% that are so bright their light overwhelms nearby stars.
% Understanding the spectrum is useful because:
%   - Number of properties of the quasar can b
%   - Properties of the regions of the universe through
%       which light passes can be evaluated.

% Flux -> Lumens per square meter (intensity)
% Lambda -> Wavelength in Angstroms

% Lyman-alpha wavelength:
%   Wavelength beyond which intervening particles negligbly
%   interfere with light emitted from the quasar.
%   For wavelengths greater than the Lyman-alpha wavelength,
%   the observed spectrum can be modeled as a smooth spectrum
%   plus noise
%           f_obs(lambda) = f(lambda) + noise(lambda)
%   For wavelengths below the Lyman-alpha wavelength, a region
%   of the spectrum known as the Lyman-alpha forest, intervening
%   matter attenuates the observed signal. For these wavelengths,
%   we model the observed frequency as
%           f_obs(lambda) = absorp(lambda)*f(lambda) + noise(lambda)
%   Physicists want to know what the absorption function is because
%   it tells them about the distribution of the neutral hydrogen 
%   (the stuff absorping those wavelengths of light) in 
%   otherwise unreachable parts of the universe.

% Hubble Space Telescope Faint Object Spectograph
%   -> Spectra of Active Galactic Nuclei and Quasars