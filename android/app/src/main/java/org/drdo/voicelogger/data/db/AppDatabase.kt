package org.drdo.voicelogger.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import org.drdo.voicelogger.data.model.AuditLog
import org.drdo.voicelogger.data.model.LogbookEntry

@Database(
    entities = [LogbookEntry::class, AuditLog::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun logbookEntryDao(): LogbookEntryDao
    abstract fun auditLogDao(): AuditLogDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "drdo_voice_logger.db"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
