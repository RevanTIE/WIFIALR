% Datfile_convert script
[FileName,PathName,FilterIndex] = uigetfile('*.dat','MultiSelect','on'); %First cycle removed. By: Emmanuel L

csi_trace = read_bf_file(strcat(PathName,char(FileName)));
% eliminate empty cell
xx = find(cellfun('isempty', csi_trace));
csi_trace(xx) = [];

% Extract CSI information for each packet
fprintf('Have CSI for %d packets\n', length(csi_trace))

% Scaled into linear
csi = zeros(length(csi_trace),3,30);
timestamp = zeros(1,length(csi_trace));
temp = [];
for packet_index = 1:length(csi_trace)
	csi(packet_index,:,:) = get_scaled_csi(csi_trace{packet_index});
	timestamp(packet_index) = csi_trace{packet_index}.timestamp_low * 1.0e-6;
end
timestamp = timestamp';

% File export
csi_amp_matrix = permute(db(abs(squeeze(csi))), [2 3 1]);
csi_phase_matrix = permute(angle(squeeze(csi)), [2 3 1]);

for k=1:size(csi_phase_matrix,1)
	for j=1:size(csi_phase_matrix,3)
		csi_phase_matrix2(k,:,j) = phase_calibration(csi_phase_matrix(k,:,j))';
	end
end

for packet_index = 1:length(csi_trace)
	temp = [temp;horzcat(reshape(csi_amp_matrix(:,:,packet_index)',[1,90]),...
						 reshape(csi_phase_matrix2(:,:,packet_index)',[1,90]))];
end

temp(temp==-Inf) = NaN;  %-Inf values replaced by NaN. By: Emmanuel L
FileName_new = erase(FileName,".dat"); %.dat string removed from filename. By: Emmanuel L
csvwrite([char(FileName_new),'.csv'],horzcat(timestamp,temp));