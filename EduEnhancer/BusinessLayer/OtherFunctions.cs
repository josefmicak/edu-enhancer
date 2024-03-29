﻿using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using ArtificialIntelligenceTools;
using System.Runtime.InteropServices;
using System.Net.NetworkInformation;
using Microsoft.AspNetCore.Http;

namespace BusinessLayer
{
    /// <summary>
    /// Functions not related to results, templates or users (entities: GlobalSettings)
    /// </summary>
    public class OtherFunctions
    {
        private DataFunctions dataFunctions;
        private readonly IConfiguration _configuration;

        public OtherFunctions(CourseContext context, IConfiguration configuration)
        {
            dataFunctions = new DataFunctions(context);
            _configuration = configuration;
        }

        public DbSet<GlobalSettings> GetGlobalSettingsDbSet()
        {
            return dataFunctions.GetGlobalSettingsDbSet();
        }

        public GlobalSettings GetGlobalSettings()
        {
            return dataFunctions.GetGlobalSettings();
        }

        /// <summary>
        /// Checks whether the testing mode is on or off
        /// </summary>
        public bool GetTestingMode()
        {
            GlobalSettings globalSettings = GetGlobalSettings();
            return globalSettings.TestingMode;
        }

        /// <summary>
        /// In case testing mode has been previously enabled, it is automatically turned on after the application is started again
        /// </summary>
        public void InitialTestingModeSettings()
        {
            Config.TestingMode = GetTestingMode();
        }

        /// <summary>
        /// Sets the platform on which the application is running (Windows/Linux)
        /// </summary>
        public void SelectedPlatformSettings()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Config.SelectedPlatform = EnumTypes.Platform.Windows;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                Config.SelectedPlatform = EnumTypes.Platform.Linux;
            }
        }

        /// <summary>
        /// Sets testing mode (on/off)
        /// </summary>
        public async Task ChangeGlobalSettings(string testingMode)
        {
            var globalSettings = GetGlobalSettings();
            if (globalSettings != null)
            {
                if (testingMode == "testingModeOff")
                {
                    globalSettings.TestingMode = false;
                    Config.TestingMode = false;
                }
                else if (testingMode == "testingModeOn")
                {
                    globalSettings.TestingMode = true;
                    Config.TestingMode = true;
                }

                await dataFunctions.SaveChangesAsync();
            }
        }

        /// <summary>
        /// Checks whether computations performed in the Python scripts are performed via GPU or CPU
        /// </summary>
        public string GetAIDeviceName()
        {
            return PythonFunctions.GetDevice();
        }

        /// <summary>
        /// Returns Google Client ID (used for OAuth 2.0)
        /// </summary>
        public string GetGoogleClientId()
        {
            return _configuration["Authentication:Google:ClientId"];
        }

        /// <summary>
        /// Deletes testing data (data used to test artificial intelligence capabilities)
        /// </summary>
        public async Task DeleteTestingData()
        {
            await dataFunctions.DeleteTestingData();
        }
    }
}