namespace Common
{
    public static class FileIO
    {
		private static T Access<T>(object[] args, Func<object[], T> action)
        {
			// Unique id for global mutex - Global prefix means it is global to the machine
			// We use filePath to ensure the mutex is only held for the particular file
			string mutexId = string.Format("Global\\{{{0}}}", Path.GetFileNameWithoutExtension((string)args[0]));

			// We create/query the Mutex
			using (var mutex = new Mutex(false, mutexId))
			{
				var hasHandle = false;

				try
				{
					// We wait for lock to release
					hasHandle = mutex.WaitOne(Timeout.Infinite, false);

					// Perform action
					return action.Invoke(new object[] { args });
				}
				finally
				{
					// If we have the Mutex, we release it
					if (hasHandle)
					{
						mutex.ReleaseMutex();
					}
				}
			}
		}

		public static void WriteAllText(string filePath, string output)
		{
			Access(new object[] { filePath, output }, delegate (object[] args) {
				File.WriteAllText(filePath, output);
				return true;
			});
		}

		public static void AppendAllText(string filePath, string output)
		{
			Access(new object[] { filePath, output }, delegate (object[] args)
			{
				File.AppendAllText(filePath, output);
				return true;
			});
		}

		public static string[] ReadAllLines(string filePath)
		{
			if (File.Exists(filePath)) {
				return Access(new object[] { filePath }, delegate (object[] args) {
					return File.ReadAllLines(filePath);
				});
			}
			return new string[] {};
		}

		public static void Delete(string filePath)
		{
            if (File.Exists(filePath))
            {
				Access(new object[] { filePath }, delegate (object[] args) {
					File.Delete(filePath);
					return true;
				});
			}
		}
	}
}
