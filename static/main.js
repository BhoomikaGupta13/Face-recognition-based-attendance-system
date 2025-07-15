// Authentication
document.addEventListener('DOMContentLoaded', () => {
    // Login functionality
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(loginForm);
            const data = Object.fromEntries(formData.entries());
            
            try {
                const response = await axios.post('/token', 
                    `username=${data.username}&password=${data.password}&grant_type=password`,
                    {
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
                    }
                );
                
                localStorage.setItem('token', response.data.access_token);
                window.location.href = '/dashboard';
            } catch (error) {
                document.getElementById('loginError').classList.remove('hidden');
            }
        });
    }

    // Check authentication
    const token = localStorage.getItem('token');
    if (!token && window.location.pathname !== '/') {
        window.location.href = '/';
    }

    // Logout functionality
    const logoutBtns = document.querySelectorAll('#logoutBtn');
    logoutBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            localStorage.removeItem('token');
            window.location.href = '/';
        });
    });

    // Navigation
    const dashboardBtn = document.getElementById('dashboardBtn');
    if (dashboardBtn) {
        dashboardBtn.addEventListener('click', () => {
            window.location.href = '/dashboard';
        });
    }

    const attendanceViewBtn = document.getElementById('attendanceViewBtn');
    if (attendanceViewBtn) {
        attendanceViewBtn.addEventListener('click', () => {
            window.location.href = '/attendance_view';
        });
    }

    // Dashboard functionality
    if (window.location.pathname === '/dashboard') {
        initDashboard();
    }

    // Attendance page functionality
    if (window.location.pathname === '/attendance_view') {
        initAttendancePage();
    }
});

// Dashboard Functions
async function initDashboard() {
    const token = localStorage.getItem('token');
    if (!token) return;

    // Camera management
    const cameraList = document.getElementById('cameraList');
    const cameraModal = document.getElementById('cameraModal');
    const addCameraBtn = document.getElementById('addCameraBtn');
    const cancelCameraBtn = document.getElementById('cancelCameraBtn');
    const cameraForm = document.getElementById('cameraForm');
    const cameraFeed = document.getElementById('cameraFeed');
    const recentAttendance = document.getElementById('recentAttendance');
    const startCameraBtn = document.getElementById('startCameraBtn');

    let currentCameraId = null; // To store the ID of the currently selected camera
    let cameraTimeout = null; // To store the timeout for stopping the camera

    // Load cameras
    async function loadCameras() {
        try {
            const response = await axios.get('/cameras', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            cameraList.innerHTML = '';
            if (response.data.length > 0) {
                currentCameraId = response.data[0].id; // Set default to the first camera
                startCameraBtn.disabled = false; // Enable start button if cameras exist
            } else {
                startCameraBtn.disabled = true; // Disable if no cameras
            }

            response.data.forEach(camera => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <button data-id="${camera.id}" class="camera-btn w-full text-left px-2 py-1 rounded hover:bg-gray-700 flex items-center ${camera.id === currentCameraId ? 'bg-gray-700' : ''}">
                        <span class="w-3 h-3 rounded-full mr-2 ${camera.status === 'active' ? 'bg-green-500' : 'bg-gray-500'}"></span>
                        ${camera.name} (${camera.type})
                    </button>
                `;
                cameraList.appendChild(li);
            });

            // Add camera selection event
            document.querySelectorAll('.camera-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    // Remove active state from previous button
                    document.querySelectorAll('.camera-btn').forEach(b => b.classList.remove('bg-gray-700'));
                    // Add active state to clicked button
                    btn.classList.add('bg-gray-700');
                    
                    currentCameraId = btn.dataset.id;
                    // If camera is already running, stop it first to ensure a clean restart on the new feed
                    if (cameraFeed.src.includes('/video_feed/')) {
                        cameraFeed.src = 'https://placehold.co/600x400/000000/FFFFFF?text=Camera+Feed+Stopped';
                        startCameraBtn.disabled = false;
                        if (cameraTimeout) {
                            clearTimeout(cameraTimeout);
                            cameraTimeout = null;
                        }
                    }
                    startCameraBtn.disabled = false; // Enable start button if a camera is selected
                });
            });
        } catch (error) {
            console.error('Error loading cameras:', error);
        }
    }

    // Camera modal
    addCameraBtn.addEventListener('click', () => {
        cameraModal.classList.remove('hidden');
    });

    cancelCameraBtn.addEventListener('click', () => {
        cameraModal.classList.add('hidden');
    });

    cameraForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(cameraForm);
        const data = Object.fromEntries(formData.entries());
        
        try {
            await axios.post('/cameras', {
                id: Date.now(), // Generate unique ID
                name: data.name,
                type: data.type,
                source: data.source
            }, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            cameraModal.classList.add('hidden');
            loadCameras();
        } catch (error) {
            console.error('Error adding camera:', error);
        }
    });

    // Start Camera functionality
    startCameraBtn.addEventListener('click', () => {
        if (currentCameraId) {
            cameraFeed.src = `/video_feed/${currentCameraId}`;
            startCameraBtn.disabled = true;

            // Set a timeout to stop the camera after 30 seconds
            if (cameraTimeout) {
                clearTimeout(cameraTimeout); // Clear any existing timeout
            }
            cameraTimeout = setTimeout(() => {
                cameraFeed.src = 'https://placehold.co/600x400/000000/FFFFFF?text=Camera+Feed+Stopped';
                startCameraBtn.disabled = false;
                cameraTimeout = null;
            }, 30000); // 30 seconds
        } else {
            console.warn('No camera selected to start.');
            // You might want to display a message to the user here
        }
    });

    // Load recent attendance
    async function loadRecentAttendance() {
        try {
            // Fetch only the latest record for each user within the last 2 minutes
            const response = await axios.get('/attendance', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            recentAttendance.innerHTML = '';
            if (response.data.length === 0) {
                recentAttendance.innerHTML = '<li class="p-2 text-gray-400">No recent attendance records.</li>';
            } else {
                response.data.forEach(record => {
                    const li = document.createElement('li');
                    li.className = 'bg-gray-700 p-2 rounded';
                    li.innerHTML = `
                        <div class="flex justify-between">
                            <span class="font-medium">${record.user_id}</span>
                            <span class="text-xs text-gray-400">
                                ${new Date(record.timestamp).toLocaleTimeString()}
                            </span>
                        </div>
                        <div class="text-sm">${record.direction === 'in' ? 'Time In' : 'Time Out'}</div>
                    `;
                    recentAttendance.appendChild(li);
                });
            }
        } catch (error) {
            console.error('Error loading recent attendance:', error);
            recentAttendance.innerHTML = '<li class="p-2 text-red-400">Error loading records.</li>';
        }
    }

    // Initialize dashboard
    loadCameras();
    loadRecentAttendance();
    setInterval(loadRecentAttendance, 5000); // Refresh every 5 seconds for recent attendance
}

// Attendance Page Functions
async function initAttendancePage() {
    const token = localStorage.getItem('token');
    if (!token) return;

    const tableBody = document.getElementById('attendanceTableBody');
    const applyFiltersBtn = document.getElementById('applyFiltersBtn');
    
    // Load attendance with filters
    async function loadAttendance() {
        const userId = document.getElementById('userIdFilter').value;
        const startDate = document.getElementById('startDateFilter').value;
        const endDate = document.getElementById('endDateFilter').value;
        const direction = document.getElementById('directionFilter').value;
        
        let url = '/attendance?';
        if (userId) url += `user_id=${userId}&`;
        if (startDate) url += `start_date=${startDate}&`;
        if (endDate) url += `end_date=${endDate}&`;
        if (direction) url += `direction=${direction}`;
        
        try {
            const response = await axios.get(url, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            tableBody.innerHTML = '';
            if (response.data.length === 0) {
                const row = document.createElement('tr');
                row.innerHTML = `<td colspan="3" class="px-6 py-4 text-center text-gray-500">No attendance records found.</td>`;
                tableBody.appendChild(row);
            } else {
                response.data.forEach(record => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="px-6 py-4 whitespace-nowrap">${record.user_id}</td>
                        <td class="px-6 py-4 whitespace-nowrap">${new Date(record.timestamp).toLocaleString()}</td>
                        <td class="px-6 py-4 whitespace-nowrap">${record.direction === 'in' ? 'Time In' : 'Time Out'}</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        } catch (error) {
            console.error('Error loading attendance:', error);
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="3" class="px-6 py-4 text-center text-red-500">Error loading records.</td>`;
            tableBody.appendChild(row);
        }
    }
    
    applyFiltersBtn.addEventListener('click', loadAttendance);
    
    // Initialize page
    loadAttendance();
}
