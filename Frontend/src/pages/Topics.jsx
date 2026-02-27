import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Topics = () => {
    const navigate = useNavigate();
    const { user } = useUser();
    const [topics, setTopics] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const fetchTopics = async () => {
        setLoading(true);
        setError('');
        try {
            const userId = user?.id || '';
            const resp = await fetch(`${API_BASE}/api/topics?user_id=${userId}`);
            if (!resp.ok) throw new Error('Failed to fetch topics');
            const data = await resp.json();
            setTopics(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (user) fetchTopics();
    }, [user]);

    const formatDate = (dateStr) => {
        if (!dateStr) return '';
        return new Date(dateStr).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric'
        });
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'completed': return '#22c55e';
            case 'processing': return '#f59e0b';
            case 'failed': return '#ef4444';
            default: return '#6b7280';
        }
    };

    return (
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '40px 24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
                <div>
                    <h1 style={{ fontSize: '28px', fontWeight: '800', color: '#e0e7ff', margin: 0 }}>
                        📚 Topics
                    </h1>
                    <p style={{ color: '#9ca3af', fontSize: '14px', marginTop: '6px' }}>
                        Your playlist-based learning collections
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        onClick={() => navigate('/')}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: 'rgba(255,255,255,0.1)',
                            color: 'white',
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.15)'}
                        onMouseOut={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'}
                    >
                        ← Home
                    </button>
                    <button
                        onClick={() => navigate('/reports')}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: 'rgba(255,255,255,0.1)',
                            color: 'white',
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.15)'}
                        onMouseOut={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'}
                    >
                        📊 Reports
                    </button>
                </div>
            </div>

            {error && (
                <div style={{ padding: '12px 16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', color: '#fca5a5', marginBottom: '20px', fontSize: '14px' }}>
                    {error}
                </div>
            )}

            {loading ? (
                <div style={{ textAlign: 'center', padding: '60px', color: '#9ca3af' }}>
                    Loading topics...
                </div>
            ) : topics.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '80px 24px' }}>
                    <div style={{ fontSize: '48px', marginBottom: '16px' }}>📚</div>
                    <h3 style={{ color: '#e0e7ff', marginBottom: '8px' }}>No topics yet</h3>
                    <p style={{ color: '#9ca3af', marginBottom: '24px', fontSize: '14px' }}>
                        Process a YouTube playlist to create your first structured learning topic
                    </p>
                    <button
                        onClick={() => navigate('/')}
                        style={{
                            padding: '12px 24px',
                            background: 'linear-gradient(135deg, rgba(99,102,241,0.6), rgba(34,197,94,0.5))',
                            color: 'white',
                            border: '1px solid rgba(99,102,241,0.3)',
                            borderRadius: '10px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '600'
                        }}
                    >
                        Process a Playlist
                    </button>
                </div>
            ) : (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
                    gap: '18px'
                }}>
                    {topics.map((topic) => (
                        <div
                            key={topic.topic_id}
                            onClick={() => navigate(`/topics/${topic.topic_id}`)}
                            style={{
                                background: 'rgba(255,255,255,0.03)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '16px',
                                padding: '24px',
                                cursor: 'pointer',
                                transition: 'all 0.25s ease',
                            }}
                            onMouseOver={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)';
                                e.currentTarget.style.transform = 'translateY(-3px)';
                                e.currentTarget.style.boxShadow = '0 8px 30px rgba(99,102,241,0.1)';
                            }}
                            onMouseOut={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                            }}
                        >
                            {/* Header */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                                <span style={{
                                    fontSize: '11px',
                                    padding: '3px 10px',
                                    borderRadius: '999px',
                                    background: `${getStatusColor(topic.status)}22`,
                                    color: getStatusColor(topic.status),
                                    fontWeight: '600',
                                    border: `1px solid ${getStatusColor(topic.status)}44`
                                }}>
                                    {topic.status === 'processing' ? `Processing...` : topic.status}
                                </span>
                                <span style={{ fontSize: '11px', color: '#6b7280' }}>
                                    {formatDate(topic.created_at)}
                                </span>
                            </div>

                            {/* Title */}
                            <h3 style={{
                                fontSize: '16px',
                                fontWeight: '700',
                                color: '#e0e7ff',
                                margin: '0 0 6px 0',
                                lineHeight: '1.3'
                            }}>
                                {topic.title}
                            </h3>

                            {topic.channel && (
                                <p style={{ fontSize: '12px', color: '#6b7280', margin: '0 0 16px 0' }}>
                                    {topic.channel}
                                </p>
                            )}

                            {/* Progress */}
                            <div style={{ marginBottom: '12px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#9ca3af', marginBottom: '6px' }}>
                                    <span>{topic.completed_count || 0} of {topic.video_count} videos</span>
                                    <span>{Math.round((topic.progress || 0) * 100)}%</span>
                                </div>
                                <div style={{
                                    height: '4px',
                                    backgroundColor: 'rgba(255,255,255,0.06)',
                                    borderRadius: '999px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        height: '100%',
                                        width: `${(topic.progress || 0) * 100}%`,
                                        background: topic.status === 'completed'
                                            ? 'linear-gradient(90deg, #22c55e, #10b981)'
                                            : 'linear-gradient(90deg, #6366f1, #818cf8)',
                                        borderRadius: '999px',
                                        transition: 'width 0.5s ease'
                                    }} />
                                </div>
                            </div>

                            {/* Footer */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#6b7280' }}>
                                <span>📹 {topic.video_count} videos</span>
                                {topic.difficulty_level && (
                                    <>
                                        <span>·</span>
                                        <span style={{ textTransform: 'capitalize' }}>{topic.difficulty_level}</span>
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Topics;
