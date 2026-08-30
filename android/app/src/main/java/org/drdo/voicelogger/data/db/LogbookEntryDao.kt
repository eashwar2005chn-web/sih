package org.drdo.voicelogger.data.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow
import org.drdo.voicelogger.data.model.LogbookEntry

@Dao
interface LogbookEntryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEntry(entry: LogbookEntry)

    @Update
    suspend fun updateEntry(entry: LogbookEntry)

    @Delete
    suspend fun deleteEntry(entry: LogbookEntry)

    @Query("SELECT * FROM logbook_entries WHERE id = :id")
    suspend fun getEntryById(id: String): LogbookEntry?

    @Query("SELECT * FROM logbook_entries ORDER BY startedAt DESC")
    fun getAllEntriesFlow(): Flow<List<LogbookEntry>>

    @Query("SELECT * FROM logbook_entries ORDER BY startedAt DESC")
    suspend fun getAllEntries(): List<LogbookEntry>

    @Query("SELECT * FROM logbook_entries ORDER BY startedAt DESC LIMIT 1")
    suspend fun getLatestEntry(): LogbookEntry?

    @Query("SELECT * FROM logbook_entries WHERE syncState != 'synced' ORDER BY startedAt ASC")
    suspend fun getUnsyncedEntries(): List<LogbookEntry>

    @Query("SELECT * FROM logbook_entries WHERE syncState = 'synced' AND classification = 'NORMAL' ORDER BY startedAt ASC")
    suspend fun getOldestSyncedNormalEntries(): List<LogbookEntry>

    @Query("SELECT COUNT(*) FROM logbook_entries WHERE syncState != 'synced'")
    fun getUnsyncedCountFlow(): Flow<Int>

    @Query("SELECT SUM(sizeBytes) FROM logbook_entries")
    suspend fun getTotalStorageBytes(): Long?
}
