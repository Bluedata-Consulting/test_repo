import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

interface Schedule {
    id: string;
    time: string;
    prompt: string;
    enabled: boolean;
    user_id: string;
    avatar_id: string;
}

const Scheduler: React.FC<{ userId: string, avatarId: string }> = ({ userId, avatarId }) => {
    const [schedules, setSchedules] = useState<Schedule[]>([]);
    const [time, setTime] = useState('');
    const [prompt, setPrompt] = useState('');
    const [enabled, setEnabled] = useState(true);
    const [status, setStatus] = useState('');

    const fetchSchedules = async () => {
        try {
            const response = await fetch('http://localhost:8600/api/schedules');
            if (response.ok) {
                const data = await response.json();
                setSchedules(data);
            }
        } catch (error) {
            console.error('Error fetching schedules:', error);
        }
    };

    useEffect(() => {
        fetchSchedules();
    }, []);

    useEffect(() => {
        if (status) {
            const timer = setTimeout(() => {
                setStatus('');
            }, 3000);
            return () => clearTimeout(timer);
        }
    }, [status]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const newSchedule: Schedule = {
            id: uuidv4(),
            time,
            prompt,
            enabled,
            user_id: userId,
            avatar_id: avatarId,
        };

        try {
            const response = await fetch('http://localhost:8600/api/schedules', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newSchedule),
            });

            if (response.ok) {
                setStatus('Schedule created successfully!');
                setTime('');
                setPrompt('');
                await fetchSchedules(); // Refresh the list
            } else {
                setStatus('Failed to create schedule.');
            }
        } catch (error) {
            console.error('Error creating schedule:', error);
            setStatus('Error creating schedule.');
        }
    };

    const handleDelete = async (scheduleId: string) => {
        if (window.confirm('Are you sure you want to delete this schedule?')) {
            try {
                const response = await fetch(`http://localhost:8600/api/schedules/${scheduleId}`, {
                    method: 'DELETE',
                });

                if (response.ok) {
                    setStatus('Schedule deleted successfully!');
                    await fetchSchedules(); // Refresh the list
                } else {
                    setStatus('Failed to delete schedule.');
                }
            } catch (error) {
                console.error('Error deleting schedule:', error);
                setStatus('Error deleting schedule.');
            }
        }
    };

    return (
        <div style={{ maxWidth: 900, margin: '20px auto', padding: '20px', background: '#121829', border: '1px solid #2a3350', borderRadius: 10 }}>
            <h2 style={{ margin: 0, marginBottom: 20 }}>Create a Schedule</h2>
            {status && <p style={{ color: '#38bdf8', margin: '0 0 15px' }}>{status}</p>}
            <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 15, alignItems: 'center', marginBottom: 30 }}>
                <input type="time" value={time} onChange={(e) => setTime(e.target.value)} required style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6' }} />
                <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Enter prompt" required style={{ flex: 1, padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6' }}/>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
                    Enabled
                </label>
                <button type="submit" style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6', cursor: 'pointer' }}>Create</button>
            </form>

            <h2 style={{ marginTop: 0, marginBottom: 15 }}>Active Schedules</h2>
            <ul style={{ listStyleType: 'none', padding: 0, margin: 0 }}>
                {schedules.map((schedule) => (
                    <li key={schedule.id} style={{ marginBottom: 10, padding: '12px 15px', background: '#1a2238', border: '1px solid #2a3350', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <strong style={{ color: '#cbd5e1' }}>{schedule.time}</strong> - <span style={{ color: '#94a3b8' }}>{schedule.prompt}</span>
                        </div>
                        <button onClick={() => handleDelete(schedule.id)} style={{ background: '#ef4444', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer' }}>Delete</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default Scheduler;