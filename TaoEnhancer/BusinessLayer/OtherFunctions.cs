using Common;
using DataLayer;
using DomainModel;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using NeuralNetworkTools;
using System.Runtime.InteropServices;

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

        public bool GetTestingMode()
        {
            return GetGlobalSettings().TestingMode;
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
            else
            {
                //throw exception
            }
        }

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

        public string GetAIDeviceName()
        {
            return PythonFunctions.GetDevice();
        }

        public string GetGoogleClientId()
        {
            return _configuration["Authentication:Google:ClientId"];
        }

        public string GetCurrentUserLogin()
        {
            return Config.Application["login"];
        }

        public void SetCurrentUserLogin(string login)
        {
            Config.Application["login"] = login;
        }
    }
}