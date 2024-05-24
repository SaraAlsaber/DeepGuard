% Define Modbus parameters
ipAddress = '127.0.0.1'; 
port = 502; 
slaveID = 1; 

% Create Modbus object
m = modbus('tcpip',ipAddress, port);


% Check connection status
disp(m);


% Altering the valve and pump switches
counter = 0 ;
while strcmp(m.Status,'Connected')
    disp(counter)
    write(m,'coils',counter,1);
    counter = counter + 1;
end


% Alteing the water level
counter = 0 ;
while strcmp(m.Status,'Connected')
    disp(counter)
    write(m,'holdingregs',counter,100);
    counter = counter + 1;
end


